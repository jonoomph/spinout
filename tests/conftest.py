"""Pytest configuration shared across test suites."""

import importlib
import sys
from pathlib import Path
import types

# Ensure the repository root (containing ``src``) is importable regardless of
# how the tests are invoked.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Provide a lightweight alias so both ``sim`` and ``src.sim`` resolve to the
# same module instance.  The interactive game used to import ``sim`` directly
# while the refactor switched to ``src.sim``.
try:
    importlib.import_module("sim")
except ModuleNotFoundError:
    src_sim = importlib.import_module("src.sim")
    alias = types.ModuleType("sim")
    alias.__dict__.update(src_sim.__dict__)
    sys.modules["sim"] = alias
