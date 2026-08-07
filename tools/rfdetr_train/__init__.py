"""Headless RF-DETR Stage-1 training helpers for VM / SSH (no Colab UI)."""

from .taxonomy import CLASS_NAMES, CLASS_TO_ID, resolve_class

__all__ = ["CLASS_NAMES", "CLASS_TO_ID", "resolve_class"]
