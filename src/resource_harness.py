"""Simulated edge-hardware resource caps.

Real edge deployment validation (Docker --cpus/--memory, or a physical
Raspberry Pi 4 / Jetson Nano) is described in RESEARCH_IMPLEMENTATION_PLAN.md
Phase 4 as the stretch goal. For day-to-day development and for machines
without Docker available, this module provides an in-process measurement +
soft-cap harness:

  - CPU affinity capping via os.sched_setaffinity (restrict to N cores),
    matching the core counts of the target devices (RESOURCE_TIERS).
  - Peak memory measurement via resource.getrusage(RLIMIT_AS peak, ru_maxrss).
  - A hard memory ceiling via resource.setrlimit(RLIMIT_AS, ...) that raises
    MemoryError if a run exceeds the simulated device's RAM budget -- this
    is what actually makes KernelSHAP's blowup on low tiers a *measured
    failure* (timeout/OOM rate) rather than just a slow number.

Docker-based capping (recommended for the final reported results, since it
also isolates page-cache effects) is documented in
docker/resource_cap.md alongside a runner script.
"""
from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import resource
import time
import traceback
from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceTier:
    name: str
    cpu_cores: int
    memory_mb: int
    reference_device: str


# Calibrated against published specs used elsewhere in the surveyed
# literature (e.g. Raspberry Pi 4B 4GB / Jetson Nano 4GB / a generic
# "unconstrained dev laptop" upper bound for comparison).
#
# tier_pi4's memory_mb is intentionally 1536 here, not the literal 1024 a
# real Pi 4 (1GB variant) would give a process -- in development, a literal
# 1024MB RLIMIT_AS cap sits right at (sometimes under) the baseline virtual
# memory footprint of a fresh interpreter with numpy+lightgbm+shap+torch
# imported, which caused native allocators (OpenBLAS, libgomp) to enter
# unreliable near-ceiling failure modes (multi-second retry storms,
# `pthread_create` EAGAIN from libgomp trying to size a thread pool while
# already at the virtual-memory ceiling) rather than raising a clean,
# fast Python MemoryError. That nondeterminism is a property of emulating
# a hard cgroup-style memory limit via RLIMIT_AS in a bare process, not a
# bug in this harness -- see docker/resource_cap.md, which is the
# authoritative path for literal 1GB-budget numbers (Docker/cgroups use
# the kernel OOM killer, which fails fast and deterministically instead of
# racing the allocator). This in-process tier exists for iterating on the
# pipeline and for *relative* timing/memory comparisons across explainers
# under a still-tight-but-survivable budget.
RESOURCE_TIERS = {
    "tier_pi4": ResourceTier("tier_pi4", cpu_cores=4, memory_mb=1536, reference_device="Raspberry Pi 4B (4GB, 1GB budget -- see note above; use Docker for literal 1GB numbers)"),
    "tier_jetson_nano": ResourceTier("tier_jetson_nano", cpu_cores=4, memory_mb=2560, reference_device="Jetson Nano (4GB, 2GB budget -- see note above; use Docker for literal 2GB numbers)"),
    "tier_unconstrained": ResourceTier("tier_unconstrained", cpu_cores=os.cpu_count() or 4, memory_mb=8192, reference_device="Dev laptop (reference/unconstrained)"),
}


@contextlib.contextmanager
def cpu_cap(n_cores: int):
    """Restrict this process to n_cores CPUs for the duration of the block."""
    available = list(range(os.cpu_count() or 1))
    original = None
    try:
        original = os.sched_getaffinity(0)
        target = set(available[: max(1, min(n_cores, len(available)))])
        os.sched_setaffinity(0, target)
    except (AttributeError, OSError):
        original = None  # platform doesn't support affinity (e.g. macOS)
    try:
        yield
    finally:
        if original is not None:
            os.sched_setaffinity(0, original)


@contextlib.contextmanager
def memory_cap(memory_mb: int):
    """Soft RLIMIT_AS cap -- raises MemoryError if exceeded during the block.
    Restores the previous limit afterward.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    new_limit = memory_mb * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (new_limit, hard))
    except (ValueError, OSError):
        pass  # some environments (containers) disallow raising/lowering
    try:
        yield
    finally:
        try:
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
        except (ValueError, OSError):
            pass


def peak_rss_mb() -> float:
    """Peak resident set size of this process so far, in MB (Linux: KB units)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


@dataclass
class RunResult:
    elapsed_s: float
    peak_memory_mb: float
    oom: bool
    error: str | None = None


def _child_entry(fn, args, kwargs, tier: ResourceTier, queue: mp.Queue):
    """Runs inside a spawned child (fresh interpreter, no inherited thread
    state) so:
      (a) a hard C-level OOM abort (observed in practice: OpenBLAS calls
          exit() directly on allocation failure rather than raising a
          catchable Python exception) only kills this child, not the whole
          experiment grid; and
      (b) explainer libraries that hold background-thread state (shap's
          KernelExplainer, tqdm's monitor thread) don't hang -- an earlier
          fork()-based version of this harness deadlocked on KernelSHAP
          specifically, because fork() only carries over the calling
          thread, so a lock held by another thread at fork time stays
          locked forever in the child. spawn avoids the whole class of bug
          at the cost of re-importing modules per call (`fn` must
          therefore be a top-level, picklable function -- see
          explainers.explain_dispatch -- not a lambda/closure).

    Also pins BLAS/OMP thread pools to 1, which is both a safety margin and
    the realistic condition on a single/few-core edge device.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    start = time.perf_counter()
    try:
        with cpu_cap(tier.cpu_cores), memory_cap(tier.memory_mb):
            result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        queue.put(("ok", result, elapsed, peak_rss_mb()))
    except MemoryError as e:
        queue.put(("oom", None, time.perf_counter() - start, peak_rss_mb(), str(e)))
    except Exception:  # noqa: BLE001 -- any failure under a cap is reportable
        queue.put(("error", None, time.perf_counter() - start, peak_rss_mb(), traceback.format_exc(limit=3)))


def run_under_tier(fn, tier: ResourceTier, *args, timeout_s: float = 60.0, **kwargs) -> tuple[object, RunResult]:
    """Run `fn(*args, **kwargs)` under the given resource tier in an
    isolated (spawned) subprocess, capturing wall time, peak memory, and
    whether it OOM'd or hard-crashed against the simulated budget. Returns
    (fn_result_or_None, RunResult). A hard crash (e.g. an unrecoverable
    native allocator abort) is recorded as oom=True rather than taking
    down the caller.

    `timeout_s` matters more than it looks: empirically, RLIMIT_AS caps
    tight enough to matter (e.g. tier_pi4's 1GB budget, which a fresh
    interpreter with numpy+lightgbm+shap already consumes most of just by
    importing) don't always fail fast. glibc malloc / OpenBLAS sometimes
    enter multi-second-to-multi-minute *retry loops* near the ceiling
    instead of raising immediately (we observed both behaviors
    nondeterministically in development -- an instant MemoryError on one
    run, a multi-minute stall on the next, for the identical call). A
    bounded timeout converts "hangs the whole grid" into "recorded as a
    timeout/OOM for this one cell," which is itself the correct
    measurement: an explainer that hits this wall on real edge hardware
    isn't usable there either. See docker/resource_cap.md for cgroup-based
    caps, which use the kernel OOM killer instead of relying on the
    allocator's own (unreliable near the ceiling) failure behavior.
    """
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_child_entry, args=(fn, args, kwargs, tier, queue))
    start = time.perf_counter()
    proc.start()

    try:
        payload = queue.get(timeout=timeout_s)
    except Exception:
        payload = None
    proc.join(timeout=5)
    if proc.is_alive():
        proc.terminate()
        proc.join()

    elapsed = time.perf_counter() - start

    if payload is None:
        # child died without reporting, or exceeded timeout_s (most likely
        # an allocator retry-storm near the memory ceiling -- see above)
        exitcode = proc.exitcode
        return None, RunResult(
            elapsed_s=elapsed,
            peak_memory_mb=float(tier.memory_mb),  # treat as having consumed the full budget
            oom=True,
            error=f"child process terminated or exceeded {timeout_s}s (exitcode={exitcode}); "
                  f"most likely a hard OOM abort or allocator retry-storm (e.g. OpenBLAS) "
                  f"under the {tier.name} memory cap",
        )

    status = payload[0]
    if status == "ok":
        _, result, elapsed_inner, mem = payload
        return result, RunResult(elapsed_s=elapsed_inner, peak_memory_mb=mem, oom=False, error=None)
    elif status == "oom":
        _, _, elapsed_inner, mem, err = payload
        return None, RunResult(elapsed_s=elapsed_inner, peak_memory_mb=mem, oom=True, error=err)
    else:
        _, _, elapsed_inner, mem, err = payload
        return None, RunResult(elapsed_s=elapsed_inner, peak_memory_mb=mem, oom=False, error=err)
