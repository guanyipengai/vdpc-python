"""Python port of the MATLAB VDPC experiment code."""

from .vdpc import (
    VDPCResult,
    avrgkneighbour,
    compute_dc,
    compute_simi,
    dbscan,
    get_distance_to_higher_density,
    get_local_density,
    mknn_clusters,
    run_vdpc,
    shapeset_to_distset,
    two_zero_least,
)

# Simple alias for convenience in other scripts
vdpc_cluster = run_vdpc

__all__ = [
    "VDPCResult",
    "avrgkneighbour",
    "compute_dc",
    "compute_simi",
    "dbscan",
    "get_distance_to_higher_density",
    "get_local_density",
    "mknn_clusters",
    "run_vdpc",
    "vdpc_cluster",
    "shapeset_to_distset",
    "two_zero_least",
]
