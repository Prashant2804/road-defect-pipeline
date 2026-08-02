"""Resolve the compute device with graceful CPU fallback."""
from __future__ import annotations

from .logging import get_logger

log = get_logger("rdd.device")


def resolve_device(requested: str = "auto", strict: bool = False) -> str:
    """Return a device string usable by ultralytics ('cuda:0' / 'cpu' / ...).

    'auto' -> cuda if available else cpu.

    An explicit `cuda` request that cannot be satisfied warns and falls back, which is
    right for a short inference run and wrong for training: a fine-tune that quietly
    drops to CPU turns a 20-minute job into a 10-hour one, and the warning scrolls past
    inside ultralytics' own banner. Callers that cannot absorb that cost pass
    strict=True and get an error instead.
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
        if strict:
            raise SystemExit(
                f"Requested {requested}, but this machine has no CUDA-capable GPU "
                f"(torch was built without CUDA, or no GPU is present).\n"
                f"  Training on CPU here takes hours rather than minutes, so this "
                f"stops instead of falling back.\n"
                f"  Use a GPU runtime (notebooks/colab_inference.ipynb, section 5b), "
                f"or pass --device cpu to accept the wait.")
        log.warning("Requested %s but CUDA unavailable -> falling back to cpu", requested)
        return "cpu"

    return requested
