"""Resolve the compute device with graceful CPU fallback."""
from __future__ import annotations

from .logging import get_logger

log = get_logger("rdd.device")


def resolve_device(requested: str = "auto") -> str:
    """Return a device string usable by ultralytics ('cuda:0' / 'cpu' / ...).

    'auto' -> cuda if available else cpu. An explicit cuda request that can't be
    satisfied logs a warning and falls back to cpu rather than crashing.
    """
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    if requested in ("auto", None, ""):
        dev = "cuda:0" if has_cuda else "cpu"
        log.info("Device auto-resolved to %s", dev)
        return dev

    if requested.startswith("cuda") and not has_cuda:
        log.warning("Requested %s but CUDA unavailable -> falling back to cpu", requested)
        return "cpu"

    return requested
