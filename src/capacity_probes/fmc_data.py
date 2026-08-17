from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .sequences import SequenceStore


@dataclass
class TransitionData:
    num_items: int
    sources: np.ndarray
    targets: np.ndarray
    groups: np.ndarray
    store: SequenceStore
    _seen_codes: np.ndarray | None = None
    _counts: sparse.csr_matrix | None = None

    @classmethod
    def from_store(cls, store: SequenceStore, maximum: int) -> "TransitionData":
        lengths = store.ends - store.starts
        transition_lengths = np.minimum(np.maximum(lengths - 1, 0), maximum)
        total = int(transition_lengths.sum())
        sources = np.empty(total, dtype=np.int32)
        targets = np.empty(total, dtype=np.int32)
        groups = np.empty(total, dtype=np.int32)
        cursor = 0
        for group, count in enumerate(transition_lengths):
            count = int(count)
            if not count:
                continue
            sequence = store.group_items(group)[-(count + 1) :]
            stop = cursor + count
            sources[cursor:stop] = sequence[:-1]
            targets[cursor:stop] = sequence[1:]
            groups[cursor:stop] = group
            cursor = stop
        return cls(store.num_items, sources, targets, groups, store)

    @property
    def size(self) -> int:
        return len(self.sources)

    def count_matrix(self) -> sparse.csr_matrix:
        if self._counts is None:
            self._counts = sparse.csr_matrix(
                (
                    np.ones(self.size, dtype=np.float32),
                    (self.sources, self.targets),
                ),
                shape=(self.num_items, self.num_items),
                dtype=np.float32,
            )
            self._counts.sum_duplicates()
            self._counts.sort_indices()
        return self._counts

    def seen_codes(self) -> np.ndarray:
        if self._seen_codes is None:
            lengths = self.store.ends - self.store.starts
            groups = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
            self._seen_codes = np.unique(groups * self.num_items + self.store.items)
        return self._seen_codes

    def sample_unseen(
        self, groups: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        seen = self.seen_codes()
        negatives = rng.integers(0, self.num_items, len(groups), dtype=np.int32)
        while True:
            codes = groups.astype(np.int64) * self.num_items + negatives
            positions = np.searchsorted(seen, codes)
            valid = positions < len(seen)
            blocked = np.zeros(len(groups), dtype=bool)
            blocked[valid] = seen[positions[valid]] == codes[valid]
            if not blocked.any():
                return negatives
            negatives[blocked] = rng.integers(
                0, self.num_items, int(blocked.sum()), dtype=np.int32
            )

