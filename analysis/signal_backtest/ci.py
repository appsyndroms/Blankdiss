"""CI-hjälpfunktioner för signal-backtest."""
from __future__ import annotations
import argparse
from pathlib import Path
PACKAGE_DIR = Path(
    "analysis/signal_backtest"
)
OUTPUT_PATH = Path(
    "data/processed/ml/"
    "signal_backtest.json"
)
def verify_package() -> None:
    """Verifiera signal-backtest-paketet."""
    required = [
        "__init__.py",
        "__main__.py",
        "config.py",
        "metrics.py",
        "reporting.py",
        "runner.py",
    ]
    missing = [
        name
        for name in required
        if not (
            PACKAGE_DIR / name
        ).is_file()
    ]
    if missing:
        raise SystemExit(
            "Signal backtest-paket "
            "saknar:\n"
            + "\n".join(
                f"- {name}"
                for name in missing
            )
        )
    print(
        "Signal backtest-paket finns."
    )
def verify_result() -> None:
    """Verifiera signal-backtest-resultatet."""
    if not OUTPUT_PATH.is_file():
        raise SystemExit(
            "signal_backtest.json saknas."
        )
    size_kb = (
        OUTPUT_PATH.stat().st_size
        / 1024
    )
    print("================================")
    print("SIGNAL BACKTEST KLAR")
    print("================================")
    print(
        f"Resultat: {size_kb:.1f} KB"
    )
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "verify-package",
            "verify-result",
        ],
    )
    args = parser.parse_args()
    if args.command == "verify-package":
        verify_package()
    elif args.command == "verify-result":
        verify_result()
if __name__ == "__main__":
    main()
