from __future__ import annotations

import hashlib
from io import BytesIO
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import capacity_probes.data as data_module
from capacity_probes.data import prepare_release
from scripts.build_public_dataset import extract_zip, prepare_movielens_archive


def test_movielens_zip_is_prepared_without_a_system_extractor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "data/raw/ml_fixture"
    raw.mkdir(parents=True)
    with zipfile.ZipFile(raw / "fixture.zip", "w") as bundle:
        bundle.writestr("fixture/ratings.csv", "user,item\n1,2\n")
        bundle.writestr("fixture/items.csv", "item\n2\n")
    module = SimpleNamespace(
        DATASET_NAME="ml_fixture",
        INTERACTIONS_FILENAME="ratings.csv",
        ZIP_FILENAME="fixture.zip",
        EXTRACTED_DIRNAME="fixture",
    )

    prepare_movielens_archive(module)

    assert (raw / "ratings.csv").read_text() == "user,item\n1,2\n"
    assert (raw / "items.csv").read_text() == "item\n2\n"
    assert not (raw / "fixture").exists()


def test_zip_extraction_rejects_parent_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../outside.txt", "unsafe")

    with pytest.raises(RuntimeError, match="Unsafe ZIP member"):
        extract_zip(archive, tmp_path / "destination")

    assert not (tmp_path / "outside.txt").exists()


def test_release_archives_are_manual_only(tmp_path: Path) -> None:
    manifest = tmp_path / "data" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        '{"release":{"archives":{"s3_beauty.zip":"unused"}},"processed":{}}'
    )

    with pytest.raises(
        FileNotFoundError,
        match="Obtain it manually or use --source public",
    ):
        prepare_release(tmp_path, tmp_path / "processed", ("s3_beauty",))


def test_public_download_has_a_bounded_network_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"public dataset fixture\n"
    calls = []

    def fake_urlopen(url: str, timeout: int):
        calls.append((url, timeout))
        return BytesIO(payload)

    monkeypatch.setattr(data_module.urllib.request, "urlopen", fake_urlopen)
    destination = tmp_path / "fixture.txt"
    data_module._download(
        "https://example.test/fixture.txt",
        destination,
        hashlib.sha256(payload).hexdigest(),
    )

    assert destination.read_bytes() == payload
    assert calls == [
        ("https://example.test/fixture.txt", data_module.DOWNLOAD_TIMEOUT_SECONDS)
    ]
