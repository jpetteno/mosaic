"""
PATH SETUP
==========
Adds all src/ subdirectories to sys.path so that cross-module imports
work regardless of which directory scripts are run from.

Import this at the top of any src/ file that imports from sibling modules:
    import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src._path_setup import *  # noqa

This is called automatically by run.py before anything else executes.
"""
import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC          = os.path.join(_PROJECT_ROOT, "src")

for _p in [
    _PROJECT_ROOT,          # for custom_settings.py
    _SRC,                   # for config.py, data_setup.py
    os.path.join(_SRC, "simulation"),
    os.path.join(_SRC, "spatial"),
    os.path.join(_SRC, "analysis"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
