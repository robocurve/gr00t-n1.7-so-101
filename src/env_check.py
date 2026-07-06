"""Environment smoke test, run on the actual training GPU (H100).

Checks: torch/flash-attn imports, DCGM profiling under gVisor, NVML GPM support,
base model download, GR00T model instantiation + named-module dump for LoRA
targeting, category-lora import.
"""

import json
import os
import subprocess
import sys
import time


def check(name, fn):
    try:
        out = fn()
        print(f"[OK]   {name}: {out if out is not None else ''}")
        return True, out
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        return False, None


def main():
    results = {}

    def torch_check():
        import torch

        assert torch.cuda.is_available()
        return f"torch {torch.__version__}, cuda {torch.version.cuda}, dev {torch.cuda.get_device_name(0)}"

    results["torch"], _ = check("torch+cuda", torch_check)

    def fa_check():
        import flash_attn

        return f"flash_attn {flash_attn.__version__}"

    results["flash_attn"], _ = check("flash-attn", fa_check)

    def torchcodec_check():
        import torchcodec  # noqa: F401

        return "importable"

    results["torchcodec"], _ = check("torchcodec", torchcodec_check)

    def category_lora_check():
        from category_lora import CategoryLoRAConfig, wrap_in_place  # noqa: F401

        return "importable"

    results["category_lora"], _ = check("category-lora", category_lora_check)

    # --- DCGM profiling under gVisor ---
    def dcgm_check():
        subprocess.run(["nv-hostengine", "-b", "127.0.0.1"], check=True, timeout=60)
        time.sleep(2)
        out = subprocess.run(
            ["dcgmi", "dmon", "-e", "1004,1005", "-c", "3", "-d", "1000"],
            capture_output=True, text=True, timeout=120,
        )
        print(out.stdout[-2000:])
        print(out.stderr[-1000:])
        assert out.returncode == 0, f"dcgmi rc={out.returncode}"
        # values may be N/A on idle GPU; success = fields readable at all
        assert "1004" in out.stdout or "TENSO" in out.stdout.upper() or out.stdout.strip()
        return "dcgmi dmon 1004/1005 readable"

    results["dcgm"], _ = check("DCGM profiling (gVisor)", dcgm_check)

    # --- NVML GPM (fallback tier 2) ---
    def gpm_check():
        import pynvml as nv

        nv.nvmlInit()
        h = nv.nvmlDeviceGetHandleByIndex(0)
        sup = nv.nvmlGpmQueryDeviceSupport(h)
        assert sup.isSupportedDevice, "GPM not supported"
        return "GPM supported"

    results["gpm"], _ = check("NVML GPM", gpm_check)

    # --- our sampler thread end-to-end (no wandb: collect locally) ---
    def sampler_check():
        import threading

        import torch

        sys.path.insert(0, "/root/proj/src")
        from gpu_metrics import GpuMetricsSampler

        sampler = GpuMetricsSampler(interval_s=2.0, sink="print")
        sampler.start()
        # busy loop of bf16 matmuls to light up tensor cores
        a = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
        t0 = time.time()
        while time.time() - t0 < 8:
            a = a @ a
            a = a / a.norm()
        torch.cuda.synchronize()
        sampler.stop()
        assert sampler.samples_taken >= 2, f"only {sampler.samples_taken} samples"
        return f"source={sampler.source}, samples={sampler.samples_taken}, last={sampler.last_values}"

    results["sampler"], _ = check("gpu_metrics sampler", sampler_check)

    # --- base model download + instantiation + module dump ---
    def model_check():
        from huggingface_hub import snapshot_download

        path = snapshot_download("nvidia/GR00T-N1.7-3B")
        return path

    results["model_download"], model_path = check("base model download", model_check)

    def module_dump():
        import torch
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7

        model = Gr00tN1d7.from_pretrained("nvidia/GR00T-N1.7-3B", torch_dtype=torch.bfloat16)
        lines = []
        for name, mod in model.named_modules():
            cls = type(mod).__name__
            if cls in ("Linear", "CategorySpecificLinear", "CategorySpecificMLP",
                       "MultiEmbodimentActionEncoder") or "action" in name.lower():
                shape = ""
                if hasattr(mod, "weight") and hasattr(mod.weight, "shape"):
                    shape = str(tuple(mod.weight.shape))
                lines.append(f"{name}  {cls}  {shape}")
        dump = "\n".join(lines)
        os.makedirs("/data/env_check", exist_ok=True)
        with open("/data/env_check/named_modules.txt", "w") as f:
            f.write(dump)
        n_params = sum(p.numel() for p in model.parameters())
        print(dump[:8000])
        return f"{n_params/1e9:.2f}B params; module dump -> /data/env_check/named_modules.txt ({len(lines)} lines)"

    results["model_instantiate"], _ = check("GR00T instantiation + module dump", module_dump)

    with open("/data/env_check/results.json", "w") as f:
        json.dump({k: bool(v) for k, v in results.items()}, f, indent=2)

    hard_fail = [k for k in ("torch", "flash_attn", "torchcodec", "category_lora",
                             "model_download", "model_instantiate", "sampler")
                 if not results.get(k)]
    if hard_fail:
        print(f"HARD FAILURES: {hard_fail}")
        sys.exit(1)
    if not results.get("dcgm"):
        print("NOTE: DCGM profiling unavailable under gVisor; sampler will use GPM fallback.")
    print("ENV CHECK PASSED")


if __name__ == "__main__":
    main()
