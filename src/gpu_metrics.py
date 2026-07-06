"""GPU metrics sampler: DCGM profiling -> NVML GPM -> NVML utilization fallback.

Logs to wandb (commit=False so points attach to the trainer's next step; note:
samples from this thread have no strict step alignment with training steps).

Keys:
  tier 1 (DCGM) & tier 2 (GPM, equivalent metrics):
    dcgm/DCGM_FI_PROF_PIPE_TENSOR_ACTIVE   (0..1)
    dcgm/DCGM_FI_PROF_DRAM_ACTIVE          (0..1)
    dcgm/source                            ("dcgm" | "gpm")
  tier 3 (semantically different -> different keys):
    gpu/util, gpu/mem_util                 (0..100)

Any failure disables the sampler with a single warning; it never raises into
the training loop.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time

DCGM_BINDINGS = "/usr/share/datacenter-gpu-manager-4/bindings/python3"
FIELD_TENSOR = 1004  # DCGM_FI_PROF_PIPE_TENSOR_ACTIVE
FIELD_DRAM = 1005  # DCGM_FI_PROF_DRAM_ACTIVE


class GpuMetricsSampler:
    def __init__(self, interval_s: float = 10.0, sink: str = "wandb"):
        self.interval_s = interval_s
        self.sink = sink  # "wandb" | "print"
        self.source = None  # "dcgm" | "gpm" | "nvml-util" | None
        self.samples_taken = 0
        self.last_values: dict = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._reader = None

    # ---------- tier setup ----------
    def _try_dcgm(self) -> bool:
        try:
            # idempotent: nv-hostengine exits nonzero if already running; ignore.
            subprocess.run(
                ["nv-hostengine", "-b", "127.0.0.1"],
                capture_output=True, timeout=60, check=False,
            )
            time.sleep(1.0)
            if DCGM_BINDINGS not in sys.path:
                sys.path.append(DCGM_BINDINGS)
            from DcgmReader import DcgmReader  # type: ignore

            reader = DcgmReader(
                fieldIds=[FIELD_TENSOR, FIELD_DRAM],
                updateFrequency=int(self.interval_s * 1e6),
                hostname="127.0.0.1",
            )
            data = reader.GetLatestGpuValuesAsFieldIdDict()
            # must contain at least one GPU with real values eventually; probe twice
            time.sleep(self.interval_s if self.interval_s < 3 else 3)
            data = reader.GetLatestGpuValuesAsFieldIdDict()
            if not data:
                return False
            self._reader = reader
            self.source = "dcgm"
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[gpu_metrics] DCGM unavailable ({type(e).__name__}: {e}); trying GPM")
            return False

    def _try_gpm(self) -> bool:
        try:
            import pynvml as nv

            nv.nvmlInit()
            h = nv.nvmlDeviceGetHandleByIndex(0)
            if not nv.nvmlGpmQueryDeviceSupport(h).isSupportedDevice:
                return False
            self._nv = nv
            self._handles = [
                nv.nvmlDeviceGetHandleByIndex(i) for i in range(nv.nvmlDeviceGetCount())
            ]
            self.source = "gpm"
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[gpu_metrics] GPM unavailable ({type(e).__name__}: {e}); trying NVML util")
            return False

    def _try_nvml_util(self) -> bool:
        try:
            import pynvml as nv

            nv.nvmlInit()
            self._nv = nv
            self._handles = [
                nv.nvmlDeviceGetHandleByIndex(i) for i in range(nv.nvmlDeviceGetCount())
            ]
            self.source = "nvml-util"
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[gpu_metrics] NVML unavailable ({type(e).__name__}: {e}); metrics disabled")
            return False

    # ---------- sampling ----------
    def _sample_dcgm(self) -> dict:
        data = self._reader.GetLatestGpuValuesAsFieldIdDict()
        out = {}
        vals_t, vals_d = [], []
        for _gpu, fields in data.items():
            for fid, val in fields.items():
                v = getattr(val, "value", val)
                if not isinstance(v, (int, float)) or v < 0 or v > 1e6:
                    continue
                (vals_t if int(fid) == FIELD_TENSOR else vals_d).append(float(v))
        if vals_t:
            out["dcgm/DCGM_FI_PROF_PIPE_TENSOR_ACTIVE"] = sum(vals_t) / len(vals_t)
        if vals_d:
            out["dcgm/DCGM_FI_PROF_DRAM_ACTIVE"] = sum(vals_d) / len(vals_d)
        return out

    def _sample_gpm(self) -> dict:
        nv = self._nv
        tvals, dvals = [], []
        for h in self._handles:
            s1, s2 = nv.nvmlGpmSampleAlloc(), nv.nvmlGpmSampleAlloc()
            try:
                nv.nvmlGpmSampleGet(h, s1)
                time.sleep(1.0)
                nv.nvmlGpmSampleGet(h, s2)
                mg = nv.c_nvmlGpmMetricsGet_t()
                mg.version = nv.NVML_GPM_METRICS_GET_VERSION
                mg.numMetrics = 2
                mg.sample1, mg.sample2 = s1, s2
                mg.metrics[0].metricId = nv.NVML_GPM_METRIC_ANY_TENSOR_UTIL
                mg.metrics[1].metricId = nv.NVML_GPM_METRIC_DRAM_BW_UTIL
                nv.nvmlGpmMetricsGet(mg)
                tvals.append(mg.metrics[0].value / 100.0)  # % -> 0..1 to match DCGM
                dvals.append(mg.metrics[1].value / 100.0)
            finally:
                nv.nvmlGpmSampleFree(s1)
                nv.nvmlGpmSampleFree(s2)
        out = {}
        if tvals:
            out["dcgm/DCGM_FI_PROF_PIPE_TENSOR_ACTIVE"] = sum(tvals) / len(tvals)
        if dvals:
            out["dcgm/DCGM_FI_PROF_DRAM_ACTIVE"] = sum(dvals) / len(dvals)
        return out

    def _sample_nvml_util(self) -> dict:
        nv = self._nv
        gs, ms = [], []
        for h in self._handles:
            u = nv.nvmlDeviceGetUtilizationRates(h)
            gs.append(float(u.gpu))
            ms.append(float(u.memory))
        return {"gpu/util": sum(gs) / len(gs), "gpu/mem_util": sum(ms) / len(ms)}

    # ---------- lifecycle ----------
    def _loop(self):
        sample_fn = {
            "dcgm": self._sample_dcgm,
            "gpm": self._sample_gpm,
            "nvml-util": self._sample_nvml_util,
        }[self.source]
        failures = 0
        while not self._stop.is_set():
            try:
                vals = sample_fn()
                if vals:
                    vals["dcgm/source_tier"] = {"dcgm": 1, "gpm": 2, "nvml-util": 3}[self.source]
                    self.last_values = vals
                    self.samples_taken += 1
                    if self.sink == "wandb":
                        import wandb

                        if wandb.run is not None:
                            wandb.log(vals, commit=False)
                    else:
                        print(f"[gpu_metrics] {vals}")
                failures = 0
            except Exception as e:  # noqa: BLE001
                failures += 1
                if failures <= 2:
                    print(f"[gpu_metrics] sample failed: {type(e).__name__}: {e}")
                if failures > 10:
                    print("[gpu_metrics] too many failures; sampler disabled")
                    return
            self._stop.wait(self.interval_s)

    def start(self):
        try:
            if not (self._try_dcgm() or self._try_gpm() or self._try_nvml_util()):
                print("[gpu_metrics] no metrics source available; disabled")
                return self._stop
            print(f"[gpu_metrics] source: {self.source}")
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        except Exception as e:  # noqa: BLE001
            print(f"[gpu_metrics] start failed, disabled: {type(e).__name__}: {e}")
        return self._stop

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)


def start_gpu_metrics_thread(interval_s: float = 10.0) -> GpuMetricsSampler:
    sampler = GpuMetricsSampler(interval_s=interval_s, sink="wandb")
    sampler.start()
    return sampler
