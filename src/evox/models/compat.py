"""Read trusted pre-package EVO-X joblib artifacts without changing their files."""
from __future__ import annotations

import importlib
import sys
import threading
import types
from contextlib import contextmanager

_LOAD_LOCK = threading.RLock()


@contextmanager
def legacy_model_modules():
    """Temporarily resolve the original src.ids_model pickle class path."""
    with _LOAD_LOCK:
        aliases = {"src.ids_model": importlib.import_module("evox.models.ids")}
        parent = types.ModuleType("src")
        parent.__path__ = []
        aliases["src"] = parent
        previous = {name: sys.modules.get(name) for name in aliases}
        try:
            for name, module in aliases.items():
                if previous[name] is None:
                    sys.modules[name] = module
            yield
        finally:
            for name, module in aliases.items():
                if previous[name] is None and sys.modules.get(name) is module:
                    del sys.modules[name]
