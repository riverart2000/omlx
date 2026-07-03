"""Auto-imported at interpreter startup (Python site machinery).

Memory safety net for LongCat-Video runs launched by the omlx-video service.
By default MLX sets its memory limit to ~1.5x the device's recommended working
set, which on a 64GB machine is large enough that a big generation running
ALONGSIDE the already-loaded LLM can oversubscribe unified memory and hard-
crash / panic the whole Mac (observed: full reboot required).

We cap this process's MLX memory + cache from env vars set by the server:

  LONGCAT_MLX_MEM_LIMIT_GB   hard MLX memory limit (GB). If graph eval would
                             exceed it, MLX raises instead of exhausting RAM,
                             so the job fails gracefully rather than freezing
                             the machine.
  LONGCAT_MLX_CACHE_LIMIT_GB free-cache ceiling (GB). Kept small so freed
                             buffers are reclaimed to the OS between segments
                             instead of being retained.

Both are best-effort: any failure here must never break the run outright, so
everything is wrapped defensively.
"""
import os


def _apply_mlx_limits():
    try:
        import mlx.core as mx
    except Exception:
        return
    GB = 1024 ** 3
    try:
        mem_gb = float(os.environ.get("LONGCAT_MLX_MEM_LIMIT_GB", "") or 0)
        if mem_gb > 0:
            mx.set_memory_limit(int(mem_gb * GB))
    except Exception:
        pass
    try:
        cache_gb = float(os.environ.get("LONGCAT_MLX_CACHE_LIMIT_GB", "") or 0)
        # An explicit "0" disables the cache entirely (max reclaim, some perf
        # cost). A positive value caps retained free buffers.
        if "LONGCAT_MLX_CACHE_LIMIT_GB" in os.environ:
            mx.set_cache_limit(int(cache_gb * GB))
    except Exception:
        pass


_apply_mlx_limits()
