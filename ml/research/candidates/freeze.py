from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ml.research.candidates.spec import (
    CandidateSpec,
    load_candidate,
)
from ml.research.candidates.verification import (
    candidate_fingerprint,
)


def _thaw(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _thaw(item)
            for key, item in value.items()
        }

    if isinstance(value, tuple):
        return [
            _thaw(item)
            for item in value
        ]

    return value


def _load_raw_candidate(
    path: Path,
) -> dict[str, Any]:
    payload = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(payload, dict):
        raise ValueError(
            f"Candidate måste vara ett objekt: {path}"
        )

    return payload


def _freeze_payload(
    candidate: CandidateSpec,
    raw_payload: dict[str, Any],
) -> dict[str, Any]:
    payload = dict(raw_payload)

    payload["candidate_status"] = "frozen"

    payload["fingerprint"] = {
        "algorithm": "sha256",
        "value": candidate_fingerprint(
            candidate
        ),
    }

    return payload


def freeze_candidate(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    freeze_at: str | None = None,
) -> Path:
    input_path = Path(input_path)

    raw_payload = _load_raw_candidate(
        input_path
    )

    candidate = load_candidate(
        input_path
    )

    if candidate.status == "frozen":
        raise ValueError(
            "Candidate är redan frozen. "
            "En frozen candidate får inte "
            "frysas om eller skrivas över."
        )

    if candidate.status == "retired":
        raise ValueError(
            "En retired candidate kan inte "
            "frysas."
        )

    if freeze_at is not None:
        raw_payload["freeze_at"] = freeze_at

        temporary_path = (
            input_path.parent
            / (
                f".{input_path.stem}"
                ".freeze_tmp.yaml"
            )
        )

        try:
            temporary_path.write_text(
                yaml.safe_dump(
                    raw_payload,
                    allow_unicode=True,
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            candidate = load_candidate(
                temporary_path
            )
        finally:
            temporary_path.unlink(
                missing_ok=True
            )

    if (
        candidate.freeze_at
        <= candidate.discovery_cutoff
    ):
        raise ValueError(
            "freeze_at måste ligga efter "
            "discovery_cutoff."
        )

    payload = _freeze_payload(
        candidate,
        raw_payload,
    )

    if output_path is None:
        output_path = (
            input_path.with_name(
                f"{input_path.stem}.frozen.yaml"
            )
        )
    else:
        output_path = Path(output_path)

    output_path = Path(output_path)

    if output_path.exists():
        raise FileExistsError(
            "Freeze-resultatet finns redan och "
            "får inte skrivas över: "
            f"{output_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        yaml.safe_dump(
            _thaw(payload),
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    frozen = load_candidate(
        output_path
    )

    actual = candidate_fingerprint(
        frozen
    )

    if frozen.status != "frozen":
        raise ValueError(
            "Freeze-verifiering misslyckades: "
            "candidate är inte frozen."
        )

    if frozen.fingerprint_algorithm != "sha256":
        raise ValueError(
            "Freeze-verifiering misslyckades: "
            "fingerprint algorithm är inte sha256."
        )

    if frozen.fingerprint_value != actual:
        raise ValueError(
            "Freeze-verifiering misslyckades: "
            "fingerprint matchar inte definitionen."
        )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a Blankdiss candidate."
        )
    )

    parser.add_argument(
        "candidate",
        help="Path till draft candidate YAML.",
    )

    parser.add_argument(
        "--output",
        help=(
            "Path till frozen candidate YAML. "
            "Default: <candidate>.frozen.yaml"
        ),
    )

    parser.add_argument(
        "--freeze-at",
        help=(
            "Override freeze_at. "
            "Format: ISO-8601 med timezone."
        ),
    )

    args = parser.parse_args()

    output = freeze_candidate(
        args.candidate,
        args.output,
        freeze_at=args.freeze_at,
    )

    print(
        f"Candidate frozen: {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
