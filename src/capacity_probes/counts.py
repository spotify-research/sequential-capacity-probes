from __future__ import annotations

import gc
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy import sparse

from .sequences import SequenceStore


def _slug(spec: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", spec)


def _store_identity(store: SequenceStore) -> dict:
    digest = hashlib.sha256()
    for values in (store.catalog, store.users, store.items):
        digest.update(np.ascontiguousarray(values).view(np.uint8))
    return {
        "sha256": digest.hexdigest(),
        "items": store.num_items,
        "events": len(store.items),
    }


class OffsetCache:
    def __init__(self, directory: Path, store: SequenceStore):
        self.directory = directory
        self.store = store
        directory.mkdir(parents=True, exist_ok=True)
        identity_path = directory / "identity.json"
        expected = _store_identity(store)
        if identity_path.exists() and json.loads(identity_path.read_text()) != expected:
            raise ValueError(f"Cache belongs to different training data: {directory}")
        identity_path.write_text(json.dumps(expected, sort_keys=True) + "\n")

    def offset_path(self, offset: int) -> Path:
        return self.directory / f"offset_{offset:02d}.npz"

    def combined_path(self, spec: str) -> Path:
        return self.directory / f"combined_{_slug(spec)}.npz"

    def build(self, max_offset: int) -> None:
        users, items = self.store.users, self.store.items
        shape = (self.store.num_items, self.store.num_items)
        for offset in range(1, max_offset + 1):
            path = self.offset_path(offset)
            if path.exists():
                matrix = sparse.load_npz(path)
                if matrix.shape != shape:
                    raise ValueError(f"Bad cache shape {matrix.shape}: {path}")
                continue
            same_user = users[:-offset] == users[offset:]
            source = items[:-offset][same_user]
            target = items[offset:][same_user]
            matrix = sparse.coo_matrix(
                (np.ones(len(source), dtype=np.float32), (source, target)),
                shape=shape,
                dtype=np.float32,
            ).tocsr()
            matrix.sum_duplicates()
            matrix.sort_indices()
            sparse.save_npz(path, matrix, compressed=False)
            print(f"COUNT offset={offset} events={len(source)} nnz={matrix.nnz}", flush=True)
            del same_user, source, target, matrix
            gc.collect()

    def combine(self, cache_key: str, weights: np.ndarray) -> sparse.csr_matrix:
        """Combine previously built offset matrices with caller-supplied weights."""
        output = self.combined_path(cache_key)
        if output.exists():
            return sparse.load_npz(output).tocsr()
        result = None
        for offset, weight in enumerate(weights, start=1):
            matrix = sparse.load_npz(self.offset_path(offset)).tocsr()
            weighted = matrix.multiply(np.float32(weight))
            result = weighted if result is None else result + weighted
        if result is None:
            raise ValueError("At least one transition offset is required")
        result = result.tocsr().astype(np.float32)
        result.sum_duplicates()
        result.eliminate_zeros()
        result.sort_indices()
        sparse.save_npz(output, result, compressed=False)
        return result
