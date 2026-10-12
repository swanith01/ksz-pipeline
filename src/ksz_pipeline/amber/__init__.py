"""AMBER (github.com/hytrac/amber) integration: readers + unit adapter.

AMBER supplies FIELDS; every kSZ method (compute_cell, qperp_power,
coherence_decomposition) is the repo's existing, validated code.
"""
from . import io, adapter  # noqa: F401
from . import io, adapter, stitch_from_amber
