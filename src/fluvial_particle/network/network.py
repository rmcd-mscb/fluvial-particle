"""Static river-network topology, polyline geometry, and sub-reach bins."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class Network:
    """Reach topology and geometry from a provider's static arrays.

    Args:
        static: mapping with at least reach_id, to_index, is_outlet, length; optionally the polyline block
            (vertex_x, vertex_y, vertex_dist, reach_vertex_start, reach_vertex_count).
        crs_wkt: CRS of the polylines, informational.
    """

    def __init__(self, static: Mapping[str, npt.NDArray[np.generic]], crs_wkt: str = "") -> None:
        """Build the topology and geometry indices from a provider's static arrays.

        Args:
            static: mapping with at least reach_id, to_index, is_outlet, length; optionally the
                polyline block (vertex_x, vertex_y, vertex_dist, reach_vertex_start, reach_vertex_count).
            crs_wkt: CRS of the polylines, informational.
        """
        self._static = static
        self.reach_id: IntArray = np.asarray(static["reach_id"], dtype=np.int64)
        self.to_index: npt.NDArray[np.int32] = np.asarray(static["to_index"], dtype=np.int32)
        self.length: FloatArray = np.asarray(static["length"], dtype=np.float64)
        self.is_outlet: npt.NDArray[np.bool_] = self.to_index < 0
        self.n_reach: int = int(self.reach_id.size)
        self.crs_wkt = crs_wkt
        self._index_by_id: dict[int, int] = {int(r): i for i, r in enumerate(self.reach_id)}
        self._parents_ptr: IntArray | None = None
        self._parents_idx: IntArray | None = None
        self._poly_index: tuple[FloatArray, IntArray, FloatArray, FloatArray] | None = None

    # ---- ids -------------------------------------------------------------
    def index_of(self, reach_id: int) -> int:
        """Index of a reach id; raises KeyError for an unknown id."""
        return self._index_by_id[int(reach_id)]

    def index_of_many(self, reach_ids: npt.ArrayLike) -> IntArray:
        """Indices of several reach ids; raises KeyError naming the first unknown id."""
        return np.array([self.index_of(r) for r in np.asarray(reach_ids).ravel()], dtype=np.int64)

    def id_of(self, index: int) -> int:
        """Reach id at an index."""
        return int(self.reach_id[index])

    # ---- topology ----------------------------------------------------------
    def _build_parents(self) -> tuple[IntArray, IntArray]:
        if self._parents_ptr is None:
            child = self.to_index.astype(np.int64)
            has = child >= 0
            order = np.argsort(child[has], kind="stable")
            src = np.nonzero(has)[0][order]
            counts = np.bincount(child[has], minlength=self.n_reach)
            self._parents_ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
            self._parents_idx = src.astype(np.int64)
        assert self._parents_idx is not None
        return self._parents_ptr, self._parents_idx

    def parents(self, index: int) -> IntArray:
        """Indices of reaches flowing directly into ``index``."""
        ptr, idx = self._build_parents()
        return idx[ptr[index] : ptr[index + 1]]

    def parents_csr(self) -> tuple[IntArray, IntArray]:
        """Parent lists in CSR form: the parents of reach ``i`` are ``idx[ptr[i] : ptr[i + 1]]``.

        Returns:
            The (ptr, idx) pair, built once and cached.
        """
        return self._build_parents()

    def headwaters(self, as_index: bool = False) -> IntArray:
        """Reaches with no upstream reach (ids by default, indices with as_index)."""
        ptr, _ = self._build_parents()
        idx = np.nonzero(np.diff(ptr) == 0)[0].astype(np.int64)
        return idx if as_index else self.reach_id[idx]

    def outlets(self, as_index: bool = False) -> IntArray:
        """Reaches with to_index == -1 (ids by default, indices with as_index)."""
        idx = np.nonzero(self.is_outlet)[0].astype(np.int64)
        return idx if as_index else self.reach_id[idx]

    def upstream_of(self, reach_id: int) -> IntArray:
        """Ids of every reach draining to ``reach_id``, inclusive, in breadth-first order."""
        ptr, idx = self._build_parents()
        seen = [self.index_of(reach_id)]
        frontier = list(seen)
        while frontier:
            nxt: list[int] = []
            for i in frontier:
                nxt.extend(int(p) for p in idx[ptr[i] : ptr[i + 1]])
            seen.extend(nxt)
            frontier = nxt
        return self.reach_id[np.array(seen, dtype=np.int64)]

    # ---- geometry ----------------------------------------------------------
    @property
    def has_polylines(self) -> bool:
        """True when the static arrays carry the polyline block."""
        return "reach_vertex_start" in self._static

    def _build_poly_index(self) -> tuple[FloatArray, IntArray, FloatArray, FloatArray]:
        if self._poly_index is None:
            start = np.asarray(self._static["reach_vertex_start"], dtype=np.int64)
            count = np.asarray(self._static["reach_vertex_count"], dtype=np.int64)
            vdist = np.asarray(self._static["vertex_dist"], dtype=np.float64)
            reach_of_vertex = np.repeat(np.arange(self.n_reach), count)
            total = np.where(count > 0, vdist[np.clip(start + count - 1, 0, vdist.size - 1)], 0.0)
            stride = float(total.max()) + 1.0 if total.size else 1.0
            offset = np.arange(self.n_reach, dtype=np.float64) * stride
            # vertices of a reach are contiguous at start..start+count-1, in file order
            vertex_ids = np.concatenate([np.arange(s, s + c) for s, c in zip(start, count, strict=True)]).astype(
                np.int64
            )
            key = offset[reach_of_vertex] + vdist[vertex_ids]
            order = np.argsort(key, kind="stable")
            self._poly_index = (key[order], vertex_ids[order], total, offset)
        return self._poly_index

    def map_position(self, reach: npt.ArrayLike, s: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Map (reach, s) to x, y by scaling s / length onto the reach polyline's arc length.

        Args:
            reach: reach indices (-1 for inactive particles).
            s: distance from the upstream end (m); NaN for inactive particles.

        Returns:
            x and y arrays; NaN where there is no polyline, the reach has fewer than two vertices,
            or the particle is inactive.
        """
        r_all = np.asarray(reach, dtype=np.int64)
        s_all = np.asarray(s, dtype=np.float64)
        x = np.full(r_all.shape, np.nan)
        y = np.full(r_all.shape, np.nan)
        if not self.has_polylines or r_all.size == 0:
            return x, y
        sorted_key, sorted_vertex, total, offset = self._build_poly_index()
        start = np.asarray(self._static["reach_vertex_start"], dtype=np.int64)
        count = np.asarray(self._static["reach_vertex_count"], dtype=np.int64)
        vx = np.asarray(self._static["vertex_x"], dtype=np.float64)
        vy = np.asarray(self._static["vertex_y"], dtype=np.float64)
        vd = np.asarray(self._static["vertex_dist"], dtype=np.float64)
        ok = (r_all >= 0) & np.isfinite(s_all)
        ok[ok] &= count[r_all[ok]] >= 2
        if not ok.any():
            return x, y
        r = r_all[ok]
        frac = np.clip(s_all[ok] / self.length[r], 0.0, 1.0)
        target = frac * total[r]
        pos = np.searchsorted(sorted_key, offset[r] + target, side="right") - 1
        v0 = sorted_vertex[np.clip(pos, 0, sorted_vertex.size - 1)]
        last = start[r] + count[r] - 1
        v0 = np.clip(v0, start[r], last)
        v1 = np.minimum(v0 + 1, last)
        span = vd[v1] - vd[v0]
        w = np.where(span > 0.0, (target - vd[v0]) / np.where(span > 0.0, span, 1.0), 0.0)
        x[ok] = vx[v0] + w * (vx[v1] - vx[v0])
        y[ok] = vy[v0] + w * (vy[v1] - vy[v0])
        return x, y

    def polylines(self) -> list[tuple[FloatArray, FloatArray]]:
        """Per-reach (x, y) vertex arrays for plotting; empty when there are no polylines."""
        if not self.has_polylines:
            return []
        start = np.asarray(self._static["reach_vertex_start"], dtype=np.int64)
        count = np.asarray(self._static["reach_vertex_count"], dtype=np.int64)
        vx = np.asarray(self._static["vertex_x"], dtype=np.float64)
        vy = np.asarray(self._static["vertex_y"], dtype=np.float64)
        return [(vx[s : s + c], vy[s : s + c]) for s, c in zip(start, count, strict=True)]


class NetworkBins:
    """Nearly uniform sub-reach bins for counts and concentration.

    Each reach is split into ``ceil(length / bin_length)`` bins of equal width within the reach.

    Args:
        network: the Network to discretize.
        bin_length: target bin length (m); ``np.inf`` gives one bin per reach.
    """

    def __init__(self, network: Network, bin_length: float) -> None:
        """Initialize sub-reach bins.

        Args:
            network: the Network to discretize.
            bin_length: target bin length (m); ``np.inf`` gives one bin per reach.
        """
        self.network = network
        self.bin_length = float(bin_length)
        length = network.length
        if np.isinf(self.bin_length):
            n = np.ones(network.n_reach, dtype=np.int64)
        else:
            if self.bin_length <= 0.0:
                raise ValueError("bin_length must be positive")
            n = np.maximum(1, np.ceil(length / self.bin_length)).astype(np.int64)
        self.bins_per_reach: IntArray = n
        self.reach_bin_start: IntArray = (np.cumsum(n) - n).astype(np.int64)
        self.n_bins: int = int(n.sum())
        self.bin_reach: IntArray = np.repeat(np.arange(network.n_reach, dtype=np.int64), n)
        self._width_per_reach: FloatArray = length / n
        self.bin_width: FloatArray = np.repeat(self._width_per_reach, n)
        local = np.arange(self.n_bins, dtype=np.float64) - self.reach_bin_start[self.bin_reach]
        self.s_start: FloatArray = local * self.bin_width
        self.s_end: FloatArray = (local + 1.0) * self.bin_width

    def bin_of(self, reach: npt.ArrayLike, s: npt.ArrayLike) -> IntArray:
        """Global bin index for (reach, s); s == length maps to the reach's last bin.

        Args:
            reach: reach indices.
            s: distance from the upstream end (m); s == length maps to the reach's last bin.

        Returns:
            Global bin indices.
        """
        r = np.asarray(reach, dtype=np.int64)
        local = np.floor(np.asarray(s, dtype=np.float64) / self._width_per_reach[r]).astype(np.int64)
        local = np.clip(local, 0, self.bins_per_reach[r] - 1)
        return self.reach_bin_start[r] + local

    def midpoints_xy(self) -> tuple[FloatArray, FloatArray]:
        """Map coordinates of every bin midpoint (NaN without polylines).

        Returns:
            x and y coordinate arrays for bin midpoints.
        """
        return self.network.map_position(self.bin_reach, 0.5 * (self.s_start + self.s_end))
