"""Road Defect Detection pipeline package.

Modular stages: ingest -> preprocess -> (annotate) -> model -> (depth) ->
inference -> report. Each subpackage is runnable/testable on its own; run.py
wires them into an end-to-end CLI.
"""

__version__ = "0.1.0"
