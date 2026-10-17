"""Compatibility imports; implementation lives in electrode_structure_factors.py."""
if __package__:
    from .electrode_structure_factors import (
        ConstrictivityResult, compute_constrictivity, _distance_to, _neighbor_offsets, _periodic_pairs, _inlet_connected
    )
else:
    from electrode_structure_factors import (
        ConstrictivityResult, compute_constrictivity, _distance_to, _neighbor_offsets, _periodic_pairs, _inlet_connected
    )

__all__ = ["ConstrictivityResult", "compute_constrictivity"]
