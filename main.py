from __future__ import annotations

from pathlib import Path
import subprocess
import sys

if __package__ in (None, ""):
    package_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(package_root.parent))
    from portfolio_analyzer.constants import APP_TITLE
else:
    from .constants import APP_TITLE


def main():
    app_path = Path(__file__).with_name("streamlit_app.py")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=false",
        "--browser.gatherUsageStats=false",
    ]
    print(f"Starting {APP_TITLE} in Streamlit...")
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
