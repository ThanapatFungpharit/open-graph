"""The entry point of the Android and iOS apps (see ``[tool.flet]`` in pyproject.toml)."""

from opengraph.app import main

if __name__ == "__main__":
    raise SystemExit(main())
