"""Host CPU/memory snapshot for the SGLang process tree.

Surfaces per-process and system-wide CPU/memory usage on the inference host so
benchmarking tools can correlate host load against prefill/decode phases.
Consumed by the `/host_metrics` HTTP endpoint and `bench_serving.py`'s
`--host-metrics-interval-ms` sampler.
"""

import logging
import os
import time
from typing import Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)


class HostMetricsTracker:
    """Holds warmed psutil.Process handles and produces a CPU/memory snapshot.

    Long-lived: ``cpu_percent(interval=None)`` returns the delta since the
    previous call on the *same* Process object. The first snapshot after
    construction therefore returns 0% for every process — bench clients should
    discard the first sample.
    """

    def __init__(self, child_pid_labels: Optional[Dict[int, str]] = None):
        self._procs: Dict[int, psutil.Process] = {}
        self._labels: Dict[int, str] = {}

        # Always include the current process (HTTP server / TokenizerManager).
        self._track(os.getpid(), "tokenizer")

        for pid, role in (child_pid_labels or {}).items():
            self._track(pid, role)

        # Prime system-wide cpu_percent so the next call returns a valid delta.
        psutil.cpu_percent(interval=None, percpu=False)
        psutil.cpu_percent(interval=None, percpu=True)
        self._num_cores = psutil.cpu_count(logical=True) or 1

    def _track(self, pid: int, role: str) -> None:
        if pid in self._procs:
            return
        try:
            p = psutil.Process(pid)
            p.cpu_percent(interval=None)  # prime
            self._procs[pid] = p
            self._labels[pid] = role
        except psutil.Error as e:
            logger.warning(
                "HostMetricsTracker: could not track %s pid %s: %s", role, pid, e
            )

    def add_process(self, pid: int, role: str) -> None:
        """Late registration, e.g., a child spawned after server start."""
        self._track(pid, role)

    def snapshot(self) -> dict:
        sys_total = psutil.cpu_percent(interval=None, percpu=False)
        sys_per_core = psutil.cpu_percent(interval=None, percpu=True)
        vm = psutil.virtual_memory()
        try:
            load1, _, _ = psutil.getloadavg()
        except (AttributeError, OSError):
            load1 = 0.0

        processes: List[dict] = []
        sglang_cpu = 0.0
        sglang_rss = 0
        dead: List[int] = []
        for pid, p in self._procs.items():
            try:
                cpu = p.cpu_percent(interval=None)
                rss = p.memory_info().rss
                threads = p.num_threads()
            except psutil.NoSuchProcess:
                dead.append(pid)
                continue
            except psutil.Error:
                continue
            processes.append(
                {
                    "role": self._labels[pid],
                    "pid": pid,
                    "cpu_percent": cpu,
                    "rss_gb": rss / (1024**3),
                    "num_threads": threads,
                }
            )
            sglang_cpu += cpu
            sglang_rss += rss

        for pid in dead:
            self._procs.pop(pid, None)
            self._labels.pop(pid, None)

        return {
            "ts": time.time(),
            "system": {
                "cpu_percent_total": sys_total,
                "cpu_percent_per_core": sys_per_core,
                "num_cores": self._num_cores,
                "load_avg_1m": load1,
                "mem_used_gb": (vm.total - vm.available) / (1024**3),
                "mem_total_gb": vm.total / (1024**3),
            },
            "processes": processes,
            "aggregate": {
                "sglang_cpu_percent": sglang_cpu,
                "sglang_rss_gb": sglang_rss / (1024**3),
            },
        }
