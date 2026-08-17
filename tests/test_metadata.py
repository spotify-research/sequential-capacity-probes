from __future__ import annotations

import json
import re
from importlib.metadata import requires
from pathlib import Path

import tomli
import yaml
from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]


def _locked_versions() -> dict[str, str]:
    entries = {}
    for line in (ROOT / "requirements-lock.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==", 1)
        entries[name.lower().replace("_", "-")] = version
    return entries


def _locked_requirements() -> dict[str, Requirement]:
    entries = {}
    for line in (ROOT / "requirements-lock.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            requirement = Requirement(line)
            entries[requirement.name.lower().replace("_", "-")] = requirement
    return entries


def test_direct_dependencies_and_environment_are_fully_locked() -> None:
    project = tomli.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert "scripts" not in project
    requirements = [
        Requirement(value)
        for value in project["dependencies"] + project["optional-dependencies"]["test"]
    ]
    locked = _locked_versions()
    for requirement in requirements:
        name = requirement.name.lower().replace("_", "-")
        assert str(requirement.specifier) == f"=={locked[name]}"
    protocol = json.loads((ROOT / "configs/table1.json").read_text())["protocol"]
    environment = protocol["environment"]
    assert environment["python"] == "3.10.12"
    for name in ("numpy", "pandas", "scipy", "torch", "rectools"):
        assert locked[name] == environment[name]
    assert locked["pip"] == "26.2.1"
    assert locked["setuptools"] == "83.0.0"


def test_linux_cuda_dependencies_from_torch_are_explicitly_locked() -> None:
    locked = _locked_requirements()
    torch_requirements = [
        Requirement(value)
        for value in requires("torch") or []
        if "nvidia" in value.lower() or "triton" in value.lower()
    ]
    assert len(torch_requirements) == 15
    for expected in torch_requirements:
        actual = locked[expected.name.lower().replace("_", "-")]
        assert actual.specifier == expected.specifier
        assert str(actual.marker) == str(expected.marker)


def test_citation_metadata_identifies_the_official_paper_code() -> None:
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    assert citation["license"] == "Apache-2.0"
    assert citation["repository-code"] == (
        "https://github.com/spotify-research/sequential-capacity-probes"
    )
    assert citation["version"] == "1.0.0"
    assert str(citation["date-released"]) == "2026-08-17"
    assert [author["family-names"] for author in citation["authors"]] == [
        "Petrov",
        "Chandar",
        "Bennett",
        "Bouchard",
        "Lalmas",
    ]


def test_ci_actions_are_immutably_pinned() -> None:
    workflow = (ROOT / ".github/workflows/tests.yml").read_text()
    action_references = re.findall(r"uses:\s+[^@\s]+@([^\s#]+)", workflow)
    assert len(action_references) == 2
    assert all(re.fullmatch(r"[0-9a-f]{40}", value) for value in action_references)
