"""CLI alias: ``python -m tools.rfdetr_train.train``."""
from .train_stage1 import main

if __name__ == "__main__":
    raise SystemExit(main())
