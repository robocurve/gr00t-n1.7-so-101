"""GPU metrics sampler: DCGM profiling -> NVML GPM -> NVML utilization fallback.

DCGM tier runs ONE persistent `dcgmi dmon` child process (fresh dmon
invocations report N/A during watch warmup — streaming keeps watches warm,
the same way dcgm-exporter works). A reader thread parses rows continuously.

Logs to wandb (commit=False so points attach to the trainer's next step; note:
samples from this thread have no strict step alignment with training steps).

Keys:
  tier 1 (DCGM) & tier 2 (GPM, equivalent metrics):
    dcgm/DCGM_FI_PROF_PIPE_TENSOR_ACTIVE   (0..1)
    dcgm/DCGM_FI_PROF_DRAM_ACTIVE          (0..1)
    dcgm/source_tier                       (1|2)
  tier 3 (semantically different -> different keys):
    gpu/util, gpu/mem_util                 (0..100), dcgm/source_tier 3

Any failure demotes to the next tier; total failure disables sampling with a
warning. Never raises into the training loop.
"""

from __future__ import annotations

import subprocess
import threading
import time

FIELD_TENSOR = 1004  # DCGM_FI_PROF_PIPE_TENSOR_ACTIVE
FIELD_DRAM = 1005  # DCGM_FI_PROF_DRAM_ACTIVE


def _parse_dmon_line(line: str) -> tuple[float | None, float | None]:
    parts = line.split()
    # data rows: "GPU 0    0.123    0.456"; headers start with '#Entity'/'ID'
    if len(parts) >= 4 and parts[0] == "GPU":
        def num(s):
            try:
                v = float(s)
                return v if 0.0 <= v <= 1.0 else None
            except ValueError:
                return None

        return num(parts[2]), num(parts[3])
    return None, None


class GpuMetricsSampler:
    def __init__(self, interval_s: float = 10.0, sink: str = "wandb"):
        self.interval_s = interval_s
        self.sink = sink  # "wandb" | "print"
        self.source = None  # "dcgm" | "gpm" | "nvml-util" | None
        self.samples_taken = 0
        self.last_values: dict = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._dmon_proc: subprocess.Popen | None = None
        self._dmon_latest: dict = {}
        self._dmon_lock = threading.Lock()

    # ---------- tier 1: DCGM via persistent streaming dmon ----------
    def _try_dcgm(self) -> bool:
        try:
            # idempotent: nv-hostengine exits nonzero if already running; ignore.
            subprocess.run(
                ["nv-hostengine", "-b", "127.0.0.1"],
                capture_output=True, timeout=60, check=False,
            )
            time.sleep(1.0)
            delay_ms = max(1000, min(int(self.interval_s * 1000), 10000))
            self._dmon_proc = subprocess.Popen(
                ["dcgmi", "dmon", "-e", f"{FIELD_TENSOR},{FIELD_DRAM}", "-d", str(delay_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
            )

            def reader():
                for line in self._dmon_proc.stdout:  # type: ignore[union-attr]
                    t, d = _parse_dmon_line(line)
                    if t is None and d is None:
                        continue
                    with self._dmon_lock:
                        if t is not None:
                            self._dmon_latest["dcgm/DCGM_FI_PROF_PIPE_TENSOR_ACTIVE"] = t
                        if d is not None:
                            self._dmon_latest["dcgm/DCGM_FI_PROF_DRAM_ACTIVE"] = d
                        self._dmon_latest["_ts"] = time.time()

            threading.Thread(target=reader, daemon=True).start()
            # wait up to 3 intervals + warmup for a first real (non-N/A) row
            deadline = time.time() + max(15.0, 3 * self.interval_s)
            while time.time() < deadline:
                if self._dmon_proc.poll() is not None:
                    print(f"[gpu_metrics] dmon exited rc={self._dmon_proc.returncode}")
                    return False
                with self._dmon_lock:
                    if self._dmon_latest:
                        self.source = "dcgm"
                        print("[gpu_metrics] dcgm streaming produced values")
                        return True
                time.sleep(1.0)
            print("[gpu_metrics] dcgm streaming produced no values (all N/A); trying GPM")
            self._kill_dmon()
            return False
        except Exception as e:  # noqa: BLE001
            print(f"[gpu_metrics] DCGM unavailable ({type(e).__name__}: {e}); trying GPM")
            self._kill_dmon()
            return False

    def _kill_dmon(self):
        if self._dmon_proc is not None:
            try:
                self._dmon_proc.kill()
            except Exception:  # noqa: BLE001
                pass
            self._dmon_proc = None

    def _sample_dcgm(self) -> dict:
        if self._dmon_proc is None or self._dmon_proc.poll() is not None:
            raise RuntimeError("dmon process died")
        with self._dmon_lock:
            fresh = self._dmon_latest.get("_ts", 0) > time.time() - 3 * self.interval_s - 15
            vals = {k: v for k, v in self._dmon_latest.items() if not k.startswith("_")}
        return vals if fresh else {}

    # ---------- tier 2: NVML GPM ----------
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
            self._sample_gpm()  # probe an actual sample (support flag can lie under gVisor)
            self.source = "gpm"
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[gpu_metrics] GPM unavailable ({type(e).__name__}: {e}); trying NVML util")
            return False

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

    # ---------- tier 3: NVML utilization ----------
    def _try_nvml_util(self) -> bool:
        try:
            import pynvml as nv

            nv.nvmlInit()
            self._nv = nv
            self._handles = [
                nv.nvmlDeviceGetHandleByIndex(i) for i in range(nv.nvmlDeviceGetCount())
            ]
            self._sample_nvml_util()
            self.source = "nvml-util"
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[gpu_metrics] NVML unavailable ({type(e).__name__}: {e}); metrics disabled")
            return False

    def _sample_nvml_util(self) -> dict:
        nv = self._nv
        gs, ms = [], []
        for h in self._handles:
            u = nv.nvmlDeviceGetUtilizationRates(h)
            gs.append(float(u.gpu))
            ms.append(float(u.memory))
        return {"gpu/util": sum(gs) / len(gs), "gpu/mem_util": sum(ms) / len(ms)}

    # ---------- lifecycle ----------
    TIERS = ("dcgm", "gpm", "nvml-util")

    def _setup_from(self, start_tier_idx: int) -> bool:
        setups = {"dcgm": self._try_dcgm, "gpm": self._try_gpm, "nvml-util": self._try_nvml_util}
        for tier in self.TIERS[start_tier_idx:]:
            if setups[tier]():
                self.source = tier
                print(f"[gpu_metrics] source: {self.source}")
                return True
        return False

    def _loop(self):
        sample_fns = {
            "dcgm": self._sample_dcgm,
            "gpm": self._sample_gpm,
            "nvml-util": self._sample_nvml_util,
        }
        empty_streak = 0
        while not self._stop.is_set():
            try:
                vals = sample_fns[self.source]()
                if vals:
                    empty_streak = 0
                    vals["dcgm/source_tier"] = self.TIERS.index(self.source) + 1
                    self.last_values = dict(vals)
                    self.samples_taken += 1
                    if self.sink == "wandb":
                        import wandb

                        if wandb.run is not None:
                            wandb.log(vals, commit=False)
                    else:
                        print(f"[gpu_metrics] {vals}")
                else:
                    empty_streak += 1
            except Exception as e:  # noqa: BLE001
                empty_streak += 1
                if empty_streak <= 2:
                    print(f"[gpu_metrics] sample failed: {type(e).__name__}: {e}")
            if empty_streak >= 10:
                idx = self.TIERS.index(self.source) + 1
                self._kill_dmon()
                if idx >= len(self.TIERS) or not self._setup_from(idx):
                    print("[gpu_metrics] all tiers exhausted; sampler disabled")
                    return
                empty_streak = 0
            self._stop.wait(self.interval_s)

    def start(self):
        try:
            if not self._setup_from(0):
                print("[gpu_metrics] no metrics source available; disabled")
                return self._stop
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        except Exception as e:  # noqa: BLE001
            print(f"[gpu_metrics] start failed, disabled: {type(e).__name__}: {e}")
        return self._stop

    def stop(self):
        self._stop.set()
        self._kill_dmon()
        if self._thread:
            self._thread.join(timeout=10)


def start_gpu_metrics_thread(interval_s: float = 10.0) -> GpuMetricsSampler:
    sampler = GpuMetricsSampler(interval_s=interval_s, sink="wandb")
    sampler.start()
    return sampler
