from __future__ import annotations

from .config import load_discovery_config
from .engine import run_discovery
from .output import write_results


def run() -> None:
    config = load_discovery_config()

    if not config.enabled:
        print(
            "Discovery disabled.",
            flush=True,
        )
        return

    print(
        "=== Blankdiss Discovery ===",
        flush=True,
    )

    (
        data,
        candidates,
        results,
        findings,
    ) = run_discovery(
        config
    )

    from .engine import pool_results

    pooled = pool_results(
        results
    )

    run_dir = write_results(
        config=config,
        candidates=candidates,
        results=results,
        pooled=pooled,
        findings=findings,
        feature_rows=len(
            data.frame
        ),
    )

    print(
        f"Discovery complete: {run_dir}",
        flush=True,
    )


if __name__ == "__main__":
    run()
