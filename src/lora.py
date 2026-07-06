"""LoRA application for GR00T N1.7.

- peft LoRA on nn.Linear layers inside the action head / DiT (module names
  discovered by class+name patterns at runtime; the full resolved list is
  logged and stored in the summary).
- category-lora (jeqcho's lib) on CategorySpecificLinear layers, which peft
  cannot wrap (3D per-category weights, forward(x, cat_ids)).
- Backbone (LLM + visual) stays frozen. Small norm/embedding params inside the
  adapted submodules stay trainable.

DDP note: this plan trains single-GPU. If you ever scale out, category-lora's
per-category grads require find_unused_parameters=True (see its README) —
experiment.py hardcodes False, so revisit before multi-GPU.
"""

from __future__ import annotations

import re

import torch.nn as nn

# Substrings that mark backbone modules we must never touch.
BACKBONE_MARKERS = ("backbone", "vlm", "language_model", "visual", "vision", "qwen")


def apply_lora(model, r: int = 32, alpha: int = 64, dropout: float = 0.05):
    """Mutate `model` in place; returns the model. Summary in model._lora_summary."""
    from category_lora import CategoryLoRAConfig, wrap_in_place
    from peft.tuners.lora import LoraModel
    from peft import LoraConfig

    # 1) Freeze everything first.
    for p in model.parameters():
        p.requires_grad_(False)

    # 2) Find the action-head subtree (non-backbone) linear layers.
    linear_targets = []
    for name, mod in model.named_modules():
        lname = name.lower()
        if any(m in lname for m in BACKBONE_MARKERS):
            continue
        if isinstance(mod, nn.Linear):
            linear_targets.append(name)
    assert linear_targets, "no non-backbone nn.Linear layers found — module naming changed?"

    # 3) peft LoRA on those Linears (fully-specified names, no regex surprises).
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=linear_targets,
        bias="none",
    )
    # LoraModel injects adapters in place on the wrapped module tree.
    LoraModel(model, {"default": lora_config}, "default")

    # 4) category-lora on CategorySpecificLinear (3D per-category weights).
    n_cat = wrap_in_place(
        model,
        CategoryLoRAConfig(r=r, alpha=alpha, dropout=dropout),
        target_class_names=["CategorySpecificLinear"],
    )

    # 5) Keep norms inside the action head trainable (standard practice), plus
    #    any 'new embodiment' embedding rows if present.
    n_norms = 0
    for name, mod in model.named_modules():
        lname = name.lower()
        if any(m in lname for m in BACKBONE_MARKERS):
            continue
        if isinstance(mod, (nn.LayerNorm, nn.RMSNorm if hasattr(nn, "RMSNorm") else nn.LayerNorm)):
            for p in mod.parameters():
                p.requires_grad_(True)
            n_norms += 1

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    summary = {
        "lora_r": r,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "n_linear_lora_targets": len(linear_targets),
        "n_category_lora_wrapped": n_cat,
        "n_norm_modules_trainable": n_norms,
        "params_total": total,
        "params_trainable": trainable,
        "trainable_pct": 100.0 * trainable / total,
        "linear_targets_sample": linear_targets[:20],
    }
    model._lora_summary = summary
    print(
        f"[lora] wrapped {len(linear_targets)} Linears (peft) + {n_cat} CategorySpecificLinear "
        f"(category-lora); trainable {trainable/1e6:.1f}M / {total/1e9:.2f}B "
        f"({summary['trainable_pct']:.2f}%)"
    )
    assert trainable > 0, "nothing trainable after LoRA application"
    assert summary["trainable_pct"] < 20.0, "LoRA left too much trainable — check freezing"
    return model


def merge_lora(model):
    """Merge all adapters back into base weights (for publishing)."""
    from category_lora import unload_adapters

    unload_adapters(model)
    # peft merge: walk modules with merge_and_unload-like behavior
    from peft.tuners.lora import LoraLayer

    for _name, mod in list(model.named_modules()):
        if isinstance(mod, LoraLayer):
            mod.merge(safe_merge=True)
    # strip peft wrappers is optional if merge() folded weights; state_dict of
    # base modules now reflects merged weights.
    return model


def strip_lora_prefixes(state_dict: dict) -> dict:
    """Remove peft's `base_layer.` indirection from a merged state dict so keys
    match the original architecture."""
    out = {}
    for k, v in state_dict.items():
        if "lora_" in k:
            continue
        out[re.sub(r"\.base_layer\.", ".", k)] = v
    return out
