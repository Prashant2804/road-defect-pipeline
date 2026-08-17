#!/usr/bin/env python3
"""Thin CLI: download/prepare the merged drone/UAV Stage-1 COCO data."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.rfdetr_train.download_drone import main

if __name__ == "__main__":
    raise SystemExit(main())
