"""Python translation of the MATLAB VDPC experiment code.

The original scripts live in ``experiments/VDPC`` and were provided in MATLAB.
This module keeps the same algorithmic flow with NumPy and optional Matplotlib
to make it usable inside the Python project.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class VDPCResult:
    """Container for VDPC outputs."""

    labels: np.ndarray  # final labels after all refinement steps
    densities: np.ndarray  # local density (rho)
    deltas: np.ndarray  # distance to nearest higher density
    nearest_higher: np.ndarray  # index of the nearest higher density point
    dc: float  # cutoff distance used when computing density
    centers: np.ndarray  # indices (0-based) of detected cluster centers
    initial_labels: np.ndarray  # labels produced by vanilla DPC before DBSCAN/merging


def compute_simi(data: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance matrix (MATLAB ``pdist`` + ``squareform``)."""

    diff = data[:, None, :] - data[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def compute_dc(distset: np.ndarray, percent: float) -> float:
    """Select the distance cutoff dc based on a percentage of pairwise distances."""

    n = distset.shape[0]
    upper = distset[np.triu_indices(n, k=1)]
    if upper.size == 0:
        return 0.0

    # percent in MATLAB code was passed as a percentage value (0.4 -> 0.4%)
    pos = int(round(upper.size * percent / 100.0))
    pos = min(max(pos, 1), upper.size)
    return float(np.sort(upper)[pos - 1])


def get_local_density(distset: np.ndarray, dc: float) -> np.ndarray:
    """Gaussian-kernel local density."""

    if dc == 0:
        return np.zeros(distset.shape[0], dtype=float)
    gaus = np.exp(-((distset / dc) ** 2))
    return np.sum(gaus, axis=1) - 1.0  # subtract self-contribution


def get_distance_to_higher_density(
    distset: np.ndarray, rhos: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Distance to the nearest point with higher density (delta and neighbor)."""

    n = distset.shape[0]
    deltas = np.zeros(n, dtype=float)
    nneigh = np.full(n, -1, dtype=int)
    max_rho = np.max(rhos)

    for i in range(n):
        if rhos[i] == max_rho:
            continue
        mask = rhos > rhos[i]
        if not np.any(mask):
            continue
        candidates = distset[i, mask]
        deltas[i] = float(np.min(candidates))
        neigh_candidates = np.flatnonzero(mask & (distset[i] == deltas[i]))
        nneigh[i] = int(neigh_candidates[0]) if neigh_candidates.size else -1

    max_delta = deltas.max(initial=0.0)
    deltas[rhos == max_rho] = max_delta
    return deltas, nneigh


def two_zero_least(ss: Sequence[int]) -> int:
    """Port of ``twozeroleast.m``.

    Returns:
        The (1-based) position that marks where a gap of consecutive zeros starts.
        Returns 100 when no usable gap is found.
    """

    positions = [i + 1 for i, v in enumerate(ss) if v == 0]
    if len(positions) <= 1:
        return 100

    if len(positions) == 2 and positions[0] == positions[1] - 1:
        return positions[0] - 1

    start = 100
    if len(positions) > 2:
        consecutive = 0
        for i in range(len(positions) - 1):
            if positions[i + 1] == positions[i] + 1:
                consecutive += 1
                if consecutive >= 2:
                    start = positions[i] - consecutive
                    break
            else:
                start = 100
                consecutive = 0
    return start


def mknn_clusters(data: np.ndarray, k: int) -> np.ndarray:
    """Mutual-kNN graph based clustering (``mknn.m``)."""

    if data.size == 0 or k <= 0:
        return np.zeros(data.shape[0], dtype=int)

    dist = compute_simi(data)
    np.fill_diagonal(dist, np.inf)
    sorted_idx = np.argsort(dist, axis=1)[:, :k]

    ag = np.zeros((data.shape[0], data.shape[0]), dtype=int)
    for i in range(sorted_idx.shape[0]):
        for j in range(i + 1, sorted_idx.shape[0]):
            if len(np.intersect1d(sorted_idx[i], sorted_idx[j])) >= 2:
                ag[i, j] = 1

    labels = np.zeros(data.shape[0], dtype=int)
    current = 0
    for i in range(ag.shape[0]):
        for j in range(ag.shape[1]):
            if ag[i, j] == 1:
                if labels[i] == 0:
                    current += 1
                    labels[i] = current
                    labels[j] = labels[i]
                else:
                    labels[i] = current
                    labels[j] = labels[i]
    return labels


def avrgkneighbour(k: int, dist_mat: np.ndarray) -> np.ndarray:
    """Average distance of k nearest neighbours (``avrgkneigbour.m``)."""

    k = max(k, 1)
    dist = dist_mat.copy()
    np.fill_diagonal(dist, np.inf)
    sorted_dist = np.sort(dist, axis=1)[:, :k]
    return np.mean(sorted_dist, axis=1)


def shapeset_to_distset(shapeset: np.ndarray) -> np.ndarray:
    """Distance matrix that ignores the last column (labels) if present."""

    if shapeset.ndim != 2 or shapeset.shape[1] < 2:
        return compute_simi(shapeset)
    features = shapeset[:, :-1]
    return compute_simi(features)


def dbscan(
    dist_mat: np.ndarray, eps: float, min_pts: int, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Straight port of the MATLAB DBSCAN implementation."""

    n = dist_mat.shape[0]
    labels = np.full(n, -1, dtype=int)  # -1 = unclassified
    cluster_id = 1
    rng = rng or np.random.default_rng()
    visit_sequence = rng.permutation(n)

    def region_query(point: int) -> np.ndarray:
        return np.flatnonzero(dist_mat[:, point] <= eps)

    for pt in visit_sequence:
        if labels[pt] != -1:
            continue

        seeds = region_query(pt)
        if seeds.size < min_pts:
            labels[pt] = 100  # noise marker used by original code
            continue

        labels[seeds] = cluster_id
        seeds = np.setxor1d(seeds, np.array([pt]), assume_unique=False)
        while seeds.size:
            current = seeds[0]
            result = region_query(current)
            if result.size >= min_pts:
                for rp in result:
                    if labels[rp] in (-1, 100):
                        if labels[rp] == -1:
                            seeds = np.concatenate([seeds, [rp]])
                        labels[rp] = cluster_id
            seeds = np.setxor1d(seeds, np.array([current]), assume_unique=False)

        cluster_id += 1

    return labels


def _plot_decision_graph(rhos: np.ndarray, deltas: np.ndarray) -> None:
    """Optional helper to reproduce ``showDG.m``."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - plotting is optional
        return

    plt.figure()
    plt.plot(rhos, deltas, "o", markersize=3, markerfacecolor="k", markeredgecolor="k")
    plt.xlabel("rho")
    plt.ylabel("delta")
    plt.tight_layout()
    plt.show()


def run_vdpc(
    data: np.ndarray,
    percent: float = 0.4,
    min_delta: float = 3.5,
    min_rho: float = 0.0,
    plot: bool = False,
    rng: Optional[np.random.Generator] = None,
    continue_on_start_100: bool = False,
    center_rho_eps: float = 1e-12,
) -> VDPCResult:
    """End-to-end translation of ``main.m``.

    Args:
        data: Input samples as ``(n_samples, n_features)``.
        percent: Percentage used by ``compute_dc`` (0.4 in the MATLAB script means 0.4%).
        min_delta: Delta threshold when picking cluster centers.
        min_rho: Rho threshold when picking cluster centers.
        plot: When True, draws the decision graph.
        rng: Optional NumPy random generator (used by DBSCAN).
        continue_on_start_100: Force the pipeline to continue even if no low-density gap is found.
        center_rho_eps: Small epsilon added to the rho threshold so rho≈0 points can still become centers.
    """

    if data.ndim != 2:
        raise ValueError("data must be 2-D (n_samples, n_features)")

    rng = rng or np.random.default_rng()

    distset = compute_simi(data)
    dc = compute_dc(distset, percent)
    rhos = get_local_density(distset, dc)
    deltas, nneigh = get_distance_to_higher_density(distset, rhos)
    xstep = rhos.max() / 10.0 if rhos.size else 0.0

    if plot:
        _plot_decision_graph(rhos, deltas)

    # Some MATLAB runs treat extremely small/zero rhos as valid; center_rho_eps lets
    # you relax the rho filter slightly (e.g., 1e-12) to mimic that behavior.
    centers_mask = (rhos > (min_rho - center_rho_eps)) & (deltas > min_delta)
    ords = np.flatnonzero(centers_mask)
    cluster = np.zeros_like(rhos, dtype=int)
    for label, idx in enumerate(ords, start=1):
        cluster[idx] = label

    for idx in np.argsort(-rhos):
        if cluster[idx] == 0 and nneigh[idx] >= 0:
            cluster[idx] = cluster[nneigh[idx]]

    # Bin cluster centers by density.
    bins = max(int(math.ceil(rhos.max() / xstep)), 1) if xstep > 0 else 1
    centers_rho = rhos[ords]
    subspace: List[List[int]] = []
    for i in range(bins):
        low = i * xstep
        high = (i + 1) * xstep
        positions = [
            pos for pos, rho in enumerate(centers_rho) if (rho >= low and rho <= high)
        ]
        subspace.append(positions)

    ss = [1 if bucket else 0 for bucket in subspace]
    ones_positions = [i for i, v in enumerate(ss) if v == 1]
    if not ones_positions:
        return VDPCResult(
            labels=cluster.copy(),
            densities=rhos,
            deltas=deltas,
            nearest_higher=nneigh,
            dc=dc,
            centers=ords,
            initial_labels=cluster.copy(),
        )

    first1 = min(ones_positions)
    last1 = max(ones_positions)
    trimmed = ss[first1 : last1 + 1]
    start = two_zero_least(trimmed)

    # If no low-density gap, return the vanilla DPC result unless explicitly asked
    # to continue (some MATLAB runs proceed to DBSCAN anyway).
    if start == 100 and not continue_on_start_100:
        return VDPCResult(
            labels=cluster.copy(),
            densities=rhos,
            deltas=deltas,
            nearest_higher=nneigh,
            dc=dc,
            centers=ords,
            initial_labels=cluster.copy(),
        )
    elif start == 100 and continue_on_start_100:
        start = len(trimmed)  # fall back to using all available bins

    idx = np.zeros_like(rhos, dtype=int)
    silar_positions: List[int] = []
    for i in range(first1, start + first1):
        if i < len(subspace):
            silar_positions.extend(subspace[i])

    silarm: List[int] = []
    for pos in silar_positions:
        label = pos + 1  # cluster labels start at 1
        silarm.extend(np.flatnonzero(cluster == label).tolist())

    silarm = sorted(set(silarm))
    idx[silarm] = 200
    k = max(1, int(math.floor(math.sqrt(len(silarm))))) if silarm else 1

    newcl: List[int] = []
    if silarm:
        idxm = mknn_clusters(data[np.array(silarm)], k)
        if idxm.size and idxm.max() > 1:
            lengths = [np.sum(idxm == i) for i in range(1, idxm.max() + 1)]
            largest_idx = int(np.argmax(lengths)) + 1
            newcl = [silarm[i] for i, val in enumerate(idxm) if val != largest_idx]

    idx[newcl] = 0

    puri_datanum = np.flatnonzero(idx == 0)
    high_ords = np.intersect1d(puri_datanum, ords)

    if high_ords.size == 0 or puri_datanum.size == 0:
        labels = np.where(idx == 200, 0, cluster)
        return VDPCResult(
            labels=labels,
            densities=rhos,
            deltas=deltas,
            nearest_higher=nneigh,
            dc=dc,
            centers=ords,
            initial_labels=cluster.copy(),
        )

    index = cluster.copy()
    index_rho = rhos[high_ords]
    min_b = int(np.argmin(index_rho))
    max_bb = int(np.argmax(index_rho))

    cc = np.flatnonzero(index == index[high_ords[min_b]])
    anchor = int(cc[np.argmax(distset[high_ords[min_b], cc])]) if cc.size else 0
    distsort = np.sort(distset[anchor])
    distsort = distsort[distsort > 0]  # discard zero self-distance
    eps_pos = max(int(round(math.sqrt(max(len(cc), 1)))) - 1, 0)
    eps_pos = min(eps_pos, max(distsort.size - 1, 0))
    eps = float(distsort[eps_pos]) if distsort.size else 0.0

    lowneig = distset[anchor]
    min_pts = int(
        round(
            (np.sum(lowneig <= eps) + np.sum(distset[high_ords[max_bb]] <= eps)) / 2.0
        )
    )
    min_pts = max(min_pts, 1)

    distmat = compute_simi(data[puri_datanum])
    clust = dbscan(distmat, eps, min_pts, rng=rng)
    idx[puri_datanum] = clust

    # Re-assign DBSCAN noise to nearest surviving center.
    ordoutlier = [i for i, center in enumerate(ords) if idx[center] == 100]
    ords_mask = np.ones(len(ords), dtype=bool)
    for pos in ordoutlier + silar_positions:
        if 0 <= pos < len(ords_mask):
            ords_mask[pos] = False
    remaining_ords = ords[ords_mask]

    for i, label in enumerate(idx):
        if label == 100 and remaining_ords.size:
            nearest = remaining_ords[np.argmin(distset[i, remaining_ords])]
            idx[i] = idx[nearest]

    # Merge clusters whose peaks have low density into nearest high-density peaks.
    valid_clusters = [c for c in np.unique(idx) if 0 < c < 100]
    peaks: List[int] = []
    peak_rhos: List[float] = []
    for c in valid_clusters:
        points = np.flatnonzero(idx == c)
        if points.size == 0:
            continue
        local_rhos = rhos[points]
        peak = int(points[np.argmax(local_rhos)])
        peaks.append(peak)
        peak_rhos.append(float(local_rhos.max()))

    if peaks:
        sorted_indices = np.argsort(-np.array(peak_rhos))
        mean_rho = float(np.mean(peak_rhos))
        strong_count = int(np.sum(np.array(peak_rhos)[sorted_indices] >= mean_rho))

        if strong_count > 0 and len(peaks) > 2 * strong_count:
            peak_dist = compute_simi(data[np.array(peaks)])
            strong = sorted_indices[:strong_count]
            weak = sorted_indices[strong_count:]
            for wi in weak:
                target = strong[np.argmin(peak_dist[wi, strong])]
                src_label = valid_clusters[wi]
                tgt_label = valid_clusters[target]
                idx[idx == src_label] = tgt_label

    return VDPCResult(
        labels=idx.copy(),
        densities=rhos,
        deltas=deltas,
        nearest_higher=nneigh,
        dc=dc,
        centers=ords,
        initial_labels=cluster.copy(),
    )


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
    "shapeset_to_distset",
    "two_zero_least",
]
