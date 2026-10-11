#!/usr/bin/env python3
from __future__ import annotations

import electrode_reconstruction_pkg as _pkg
from electrode_reconstruction_pkg.cli import main

_public_names = getattr(
    _pkg,
    "__all__",
    [name for name in dir(_pkg) if not name.startswith("_")],
)

for _name in _public_names:
    globals()[_name] = getattr(_pkg, _name)


if __name__ == "__main__":
    raise SystemExit(main())
