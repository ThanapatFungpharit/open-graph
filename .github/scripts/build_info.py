"""Work out what a build is, for the build workflow, or refuse if it makes no sense.

Writes ``version``, ``label``, ``build_number`` and ``flet`` to ``$GITHUB_OUTPUT``. Run by
the system's Python on the runner, so it uses nothing outside the standard library.
"""

from __future__ import annotations

import os
import re
import sys
import tomllib
from pathlib import Path

_VERSION = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)")


def build_number(version: str) -> int:
    """A number that grows with every version, which Android requires of an update."""
    match = _VERSION.fullmatch(version)
    if match is None:
        raise ValueError(f"The version must look like 1.2.3, not {version!r}")
    major, minor, patch = (int(part) for part in match.groups())
    if minor >= 1000 or patch >= 1000 or major > 2099:
        raise ValueError(f"Version {version} doesn't fit in an Android version code")
    return major * 1_000_000 + minor * 1_000 + patch


def locked_version(lock: dict[str, list[dict[str, str]]], package: str) -> str:
    """The version of ``package`` that uv.lock holds."""
    for entry in lock["package"]:
        if entry["name"] == package:
            return entry["version"]
    raise ValueError(f"{package} is not in uv.lock")


def build_info(root: Path, ref_type: str, ref_name: str, sha: str) -> dict[str, str]:
    project = tomllib.loads((root / "pyproject.toml").read_text("utf-8"))["project"]
    lock = tomllib.loads((root / "uv.lock").read_text("utf-8"))
    version = project["version"]
    tag = ref_name if ref_type == "tag" else None
    if tag is not None and tag != f"v{version}":
        raise ValueError(f"Tag {tag} doesn't match the version in pyproject.toml, {version}")
    flet = locked_version(lock, "flet")
    if f"flet=={flet}" not in project["dependencies"]:
        # The app would bundle another Flet than the one the build tool's client is for.
        raise ValueError(f"pyproject.toml must pin flet=={flet}, the version in uv.lock")
    return {
        "version": version,
        "label": version if tag is not None else f"{version}-{sha[:7]}",
        "build_number": str(build_number(version)),
        "flet": flet,
    }


def main() -> int:
    try:
        info = build_info(
            Path.cwd(),
            os.environ.get("GITHUB_REF_TYPE", ""),
            os.environ.get("GITHUB_REF_NAME", ""),
            os.environ.get("GITHUB_SHA", "local"),
        )
    except ValueError as error:
        print(f"::error::{error}")
        return 1
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.writelines(f"{key}={value}\n" for key, value in info.items())
    print(info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
