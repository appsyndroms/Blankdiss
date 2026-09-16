"""CI-hjälpfunktioner för Blankdiss ML."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
ML_DIR = Path(
    "data/processed/ml"
)
LATEST_RUN = (
    ML_DIR / "latest_run.json"
)
ML_RESULTS = (
    ML_DIR / "ml_results.jsonl"
)
DIAGNOSTICS = (
    ML_DIR / "diagnostics.json"
)
ABLATION_QC = (
    ML_DIR / "ablation_qc.json"
)
RUNS_DIR = (
    ML_DIR / "runs"
)
def verify_package() -> None:
    """Verifiera ML-paketets nödvändiga filer."""
    required = [
        Path("__init__.py"),
        Path("ml/__init__.py"),
        Path("ml/config.py"),
        Path("ml/dataset.py"),
        Path("ml/models.py"),
        Path("ml/evaluate.py"),
        Path("ml/walk_forward.py"),
        Path("ml/train.py"),
        Path("ml/diagnostics.py"),
        Path("ml/ablation_qc.py"),
    ]
    missing = [
        str(path)
        for path in required
        if not path.is_file()
    ]
    if missing:
        raise SystemExit(
            "ML-paket saknar:\n"
            + "\n".join(
                f"- {path}"
                for path in missing
            )
        )
    print("ML-paket finns.")
def verify_results() -> None:
    """Verifiera resultat från ML-träningen."""
    if not LATEST_RUN.is_file():
        raise SystemExit(
            "latest_run.json saknas."
        )
    if not ML_RESULTS.is_file():
        raise SystemExit(
            "ml_results.jsonl saknas."
        )
    if not RUNS_DIR.is_dir():
        raise SystemExit(
            "ML runs-katalog saknas."
        )
    run_files = sorted(
        RUNS_DIR.glob("*.json")
    )
    result_files = [
        path
        for path in ML_DIR.iterdir()
        if path.is_file()
    ]
    print(
        "ML-resultat skapade."
    )
    print(
        f"Resultatfiler: "
        f"{len(result_files)}"
    )
    print(
        f"Runs: {len(run_files)}"
    )
def verify_diagnostics() -> None:
    """Verifiera att diagnostics skapats."""
    if not DIAGNOSTICS.is_file():
        raise SystemExit(
            "diagnostics.json saknas."
        )
    size_kb = (
        DIAGNOSTICS.stat().st_size
        / 1024
    )
    print(
        "ML-diagnostik skapad."
    )
    print(
        f"Storlek: {size_kb:.1f} KB"
    )
def verify_ablation_qc() -> None:
    """Verifiera att ablation QC skapats."""
    if not ABLATION_QC.is_file():
        raise SystemExit(
            "ablation_qc.json saknas."
        )
    data = json.loads(
        ABLATION_QC.read_text(
            encoding="utf-8"
        )
    )
    status = data.get(
        "status"
    )
    if status not in {
        "PASS",
        "WARN",
        "FAIL",
    }:
        raise SystemExit(
            "Ablation QC returnerade "
            f"okänd status: {status}"
        )
    print(
        f"ML Ablation QC: {status}"
    )
    if status == "FAIL":
        raise SystemExit(
            "ML Ablation QC failed."
        )
    if status == "WARN":
        print(
            "Ablation QC innehåller "
            "varningar — workflow fortsätter."
        )
def print_diagnostics() -> None:
    """Skriv en kompakt diagnostiköversikt."""
    data = json.loads(
        DIAGNOSTICS.read_text(
            encoding="utf-8"
        )
    )
    print()
    print("================================")
    print(
        "BLANKDISS PRICE SIGNAL "
        "DIAGNOSTICS"
    )
    print("================================")
    print(
        "Experiment: "
        f"{data.get('experiment', '?')}"
    )
    print(
        "Feature sets: "
        f"{len(data.get('feature_sets', []))}"
    )
    print(
        "Targets: "
        f"{len(data.get('targets', []))}"
    )
    print()
    for result in data.get(
        "results",
        [],
    ):
        feature_set = result.get(
            "feature_set",
            "?",
        )
        target = result.get(
            "target",
            "?",
        )
        feature_count = result.get(
            "feature_count",
            "?",
        )
        print(
            f"{feature_set} | "
            f"{target} | "
            f"features={feature_count}"
        )
        for window in result.get(
            "windows",
            [],
        ):
            window_info = window.get(
                "window",
                {},
            )
            test_end = window_info.get(
                "test_end",
                "?",
            )
            model = window.get(
                "selected_model",
                "?",
            )
            auc = window.get(
                "test_roc_auc"
            )
            rows = window.get(
                "test_rows",
                "?",
            )
            if auc is None:
                auc_text = "n/a"
            else:
                auc_text = f"{auc:.4f}"
            print(
                f"  {test_end} | "
                f"{model} | "
                f"AUC={auc_text} | "
                f"rows={rows}"
            )
            prediction_buckets = (
                window.get(
                    "prediction_buckets",
                    {},
                )
            )
            for year, bucket_data in (
                prediction_buckets.items()
            ):
                buckets = bucket_data.get(
                    "buckets",
                    [],
                )
                if not buckets:
                    continue
                top = buckets[-1]
                event_rate = top.get(
                    "event_rate"
                )
                lift_ratio = top.get(
                    "event_rate_lift_ratio"
                )
                mean_return = top.get(
                    "mean_return"
                )
                event_text = (
                    "n/a"
                    if event_rate is None
                    else f"{event_rate:.3f}"
                )
                lift_text = (
                    "n/a"
                    if lift_ratio is None
                    else f"{lift_ratio:.2f}x"
                )
                return_text = (
                    "n/a"
                    if mean_return is None
                    else f"{mean_return:+.2%}"
                )
                print(
                    f"    {year} top20% | "
                    f"event={event_text} | "
                    f"lift={lift_text} | "
                    f"return={return_text}"
                )
    print()
    print(
        "Full diagnostics finns "
        "i artifact."
    )
def print_latest_run() -> None:
    """Skriv en kort sammanfattning av senaste ML-körningen."""
    data = json.loads(
        LATEST_RUN.read_text(
            encoding="utf-8"
        )
    )
    print()
    print("================================")
    print("LATEST ML RUN")
    print("================================")
    for key in (
        "run_id",
        "created_at",
        "experiment",
    ):
        if key in data:
            print(
                f"{key}: {data[key]}"
            )
    results = data.get(
        "results"
    )
    if isinstance(
        results,
        list,
    ):
        print(
            f"results: {len(results)}"
        )
    print(
        "Full latest_run.json "
        "finns i artifact."
    )
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "verify-package",
            "verify-results",
            "verify-diagnostics",
            "verify-ablation-qc",
            "print-diagnostics",
            "print-latest-run",
        ],
    )
    args = parser.parse_args()
    commands = {
        "verify-package": verify_package,
        "verify-results": verify_results,
        "verify-diagnostics": verify_diagnostics,
        "verify-ablation-qc": verify_ablation_qc,
        "print-diagnostics": print_diagnostics,
        "print-latest-run": print_latest_run,
    }
    commands[args.command]()
if __name__ == "__main__":
    main()
