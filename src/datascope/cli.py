"""CLI entry point. The main experience is the Streamlit dashboard (`datascope app`)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click

APP_PATH = Path(__file__).parent / "app.py"


@click.group()
@click.version_option(package_name="datascope")
def main() -> None:
    """DataScope: data mining, modeling, and visualization techniques in one toolkit."""


@main.command()
def app() -> None:
    """Launch the interactive Streamlit dashboard."""
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(APP_PATH)], check=False)


if __name__ == "__main__":
    sys.exit(main())
