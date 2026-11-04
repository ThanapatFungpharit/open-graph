"""The script that checks a build before the build workflow spends minutes on it."""

from __future__ import annotations

import importlib.util
import re
import tomllib
from pathlib import Path
from types import ModuleType

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parent.parent


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_info", ROOT / ".github" / "scripts" / "build_info.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_info = _load()


def _project(tmp_path: Path, version: str = "1.2.3", pin: str = "flet==1.0.4") -> Path:
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nversion = "{version}"\ndependencies = ["{pin}", "numpy"]\n', "utf-8"
    )
    (tmp_path / "uv.lock").write_text(
        '[[package]]\nname = "flet"\nversion = "1.0.4"\n'
        '[[package]]\nname = "flet-cli"\nversion = "9.9.9"\n',
        "utf-8",
    )
    return tmp_path


def test_a_matching_tag_builds_a_release(tmp_path: Path) -> None:
    info = build_info.build_info(_project(tmp_path), "tag", "v1.2.3", "abcdef1234")
    assert info == {
        "version": "1.2.3",
        "label": "1.2.3",
        "build_number": "1002003",
        "flet": "1.0.4",
    }


def test_a_build_by_hand_is_labelled_with_its_commit(tmp_path: Path) -> None:
    info = build_info.build_info(_project(tmp_path), "branch", "master", "abcdef1234")
    assert info["label"] == "1.2.3-abcdef1"
    assert info["version"] == "1.2.3"  # what the build calls itself stays a valid version


def test_a_tag_that_disagrees_with_the_version_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="doesn't match"):
        build_info.build_info(_project(tmp_path), "tag", "v2.0.0", "abc")


def test_an_unpinned_flet_is_refused(tmp_path: Path) -> None:
    """The app would bundle another Flet than the build tool's client is for."""
    with pytest.raises(ValueError, match="must pin"):
        build_info.build_info(_project(tmp_path, pin="flet>=1.0.4"), "branch", "x", "abc")


def test_later_versions_get_higher_build_numbers() -> None:
    versions = ["0.1.0", "0.1.1", "0.2.0", "1.0.0", "1.0.999", "1.1.0", "2.0.0"]
    numbers = [build_info.build_number(version) for version in versions]
    assert numbers == sorted(numbers) and len(set(numbers)) == len(numbers)


@pytest.mark.parametrize("version", ["1.2", "1.2.3rc1", "01.2.3", "1.1000.0", "3000.0.0"])
def test_versions_that_cannot_be_a_version_code_are_refused(version: str) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        build_info.build_number(version)


def test_this_project_passes() -> None:
    info = build_info.build_info(ROOT, "branch", "master", "abcdef1234")
    assert re.fullmatch(r"\d+\.\d+\.\d+-abcdef1", info["label"])


@pytest.mark.parametrize("system", ["Android", "iOS"])
def test_mobile_builds_leave_out_what_has_no_wheels_there(system: str) -> None:
    """flet build runs pip on the build machine: sys_platform stays linux or darwin, and
    only platform_system tells what the packages are for."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["project"]
    environment = {"platform_system": system, "sys_platform": "linux"}
    installed = {
        requirement.name
        for requirement in map(Requirement, project["dependencies"])
        if requirement.marker is None or requirement.marker.evaluate(environment)
    }
    assert "gmpy2" not in installed and "flet-desktop" not in installed
    assert {"flet", "numpy", "sympy"} <= installed


def test_desktop_builds_keep_them() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["project"]
    environment = {"platform_system": "Linux", "sys_platform": "linux"}
    installed = {
        requirement.name
        for requirement in map(Requirement, project["dependencies"])
        if requirement.marker is None or requirement.marker.evaluate(environment)
    }
    assert {"gmpy2", "flet-desktop"} <= installed


def test_the_workflow_builds_on_the_python_that_ci_tests() -> None:
    workflow = (ROOT / ".github" / "workflows" / "build.yml").read_text("utf-8")
    tested = (ROOT / ".python-version").read_text("utf-8").strip()
    assert f'PYTHON_VERSION: "{tested}"' in workflow
