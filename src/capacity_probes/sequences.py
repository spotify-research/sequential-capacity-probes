from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from rectools import Columns
from scipy import sparse

@dataclass
class SequenceStore:
    catalog: np.ndarray
    users: np.ndarray
    items: np.ndarray
    group_users: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    user_to_group: dict[int, int]

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "SequenceStore":
        columns = [Columns.User, Columns.Item, Columns.Datetime, "_source_order"]
        ordered = frame[columns].sort_values(
            [Columns.User, Columns.Datetime, "_source_order"], kind="stable"
        )
        catalog = np.sort(ordered[Columns.Item].unique().astype(np.int64))
        users = ordered[Columns.User].to_numpy(dtype=np.int64, copy=True)
        external = ordered[Columns.Item].to_numpy(dtype=np.int64, copy=False)
        items = np.searchsorted(catalog, external).astype(np.int32)
        group_users, starts, counts = np.unique(
            users, return_index=True, return_counts=True
        )
        ends = starts + counts
        mapping = {int(user): i for i, user in enumerate(group_users.tolist())}
        return cls(catalog, users, items, group_users, starts, ends, mapping)

    @property
    def num_items(self) -> int:
        return len(self.catalog)

    def group_items(self, group: int) -> np.ndarray:
        return self.items[self.starts[group] : self.ends[group]]

    def item_frequency(self) -> np.ndarray:
        return np.bincount(self.items, minlength=self.num_items).astype(np.float64)


@dataclass
class EvaluationSet:
    store: SequenceStore
    users: np.ndarray
    groups: np.ndarray
    targets: tuple[np.ndarray, ...]

    @classmethod
    def from_holdout(cls, store: SequenceStore, holdout: pd.DataFrame) -> "EvaluationSet":
        users, groups, targets = [], [], []
        ordered = holdout.sort_values(
            [Columns.User, Columns.Datetime, "_source_order"], kind="stable"
        )
        for user, rows in ordered.groupby(Columns.User, sort=True):
            group = store.user_to_group.get(int(user))
            if group is None:
                continue
            external = rows[Columns.Item].to_numpy(dtype=np.int64)
            indices = np.searchsorted(store.catalog, external)
            valid = indices < store.num_items
            positions = np.flatnonzero(valid)
            valid[positions] = store.catalog[indices[positions]] == external[positions]
            internal = np.sort(indices[valid].astype(np.int32))
            if len(internal):
                users.append(int(user))
                groups.append(group)
                targets.append(internal)
        return cls(
            store,
            np.asarray(users, dtype=np.int64),
            np.asarray(groups, dtype=np.int32),
            tuple(targets),
        )

    def history_matrix(
        self,
        max_history: int,
        weight_factory: Callable[[int], np.ndarray],
    ) -> sparse.csr_matrix:
        """Build recent-first histories using model-supplied positional weights."""
        lengths = np.minimum(
            self.store.ends[self.groups] - self.store.starts[self.groups], max_history
        ).astype(np.int64)
        indptr = np.empty(len(lengths) + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(lengths, out=indptr[1:])
        indices = np.empty(indptr[-1], dtype=np.int32)
        values = np.empty(indptr[-1], dtype=np.float32)
        for row, (group, length) in enumerate(zip(self.groups, lengths)):
            begin, end = indptr[row], indptr[row + 1]
            history = self.store.group_items(int(group))[-int(length) :][::-1]
            indices[begin:end] = history
            values[begin:end] = weight_factory(int(length))
        return sparse.csr_matrix(
            (values, indices, indptr),
            shape=(len(self.users), self.store.num_items),
            dtype=np.float32,
        )
