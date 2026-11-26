"""Stand-ins for the AiiDAlab packages, so the widgets' logic can be tested where they are not installed."""

import importlib.util
import sys
import types


class Dummy:
    """Base class standing in for any class of the missing packages."""

    def __init__(self, *args, **kwargs):
        pass

    def __class_getitem__(cls, item):
        return cls


def fake_modules(monkeypatch, *names):
    """Put a fake module in `sys.modules` for each name whose top-level package is not installed.

    Any attribute of a fake module is a new `Dummy` subclass, so `from fake import Whatever` works.
    The fake modules, and the `aiidalab_qe_muon.app` ones imported with them, are removed after the test.
    """
    # decide first which packages are missing: a fake module hides the lookup of its own submodules.
    missing = {top for top in {n.split(".")[0] for n in names} if importlib.util.find_spec(top) is None}
    for name in names:
        if name.split(".")[0] not in missing:
            continue
        module = types.ModuleType(name)
        module.__path__ = []

        def getattr_(attr):
            if attr.startswith("__"):
                raise AttributeError(attr)
            return type(attr, (Dummy,), {})

        module.__getattr__ = getattr_
        monkeypatch.setitem(sys.modules, name, module)
    for name in [m for m in sys.modules if m.startswith("aiidalab_qe_muon.app")]:
        monkeypatch.delitem(sys.modules, name)
