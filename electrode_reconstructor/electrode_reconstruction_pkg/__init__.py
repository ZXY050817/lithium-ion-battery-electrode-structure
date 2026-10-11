from __future__ import annotations

from . import batch as _batch
from . import cbd as _cbd
from . import cli as _cli
from . import exporters as _exporters
from . import particles as _particles
from . import reconstructor as _reconstructor
from . import shared as _shared

_MODULES = (
    _shared,
    _particles,
    _batch,
    _cbd,
    _exporters,
    _reconstructor,
    _cli,
)

for _module in _MODULES:
    _public_names = getattr(
        _module,
        "__all__",
        [name for name in vars(_module) if not name.startswith("_")],
    )
    for _name in _public_names:
        globals()[_name] = getattr(_module, _name)
