"""Gemensamt register för långsiktiga entity-identiteter.

En entity är Blankdiss interna identitet för en juridisk/emittentmässig
enhet.

Externa identifierare som LEI och ISIN är observationer/alias och ska
inte användas som själva entity_id.

Exempel:

    ENT-000001
        ├── ISIN A
        ├── ISIN B
        ├── LEI A
        └── LEI B

Entity-registret ligger under shared eftersom flera komponenter ska
kunna använda samma långsiktiga identitet.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


SHARED_DIR = Path(__file__).resolve().parent

ENTITY_REGISTRY_PATH = (
    SHARED_DIR
    / "instrument_entities.jsonl"
)

ENTITY_ID_PATTERN = re.compile(
    r"^ENT-(\d+)$"
)

ENTITY_ID_PREFIX = "ENT-"
ENTITY_ID_WIDTH = 6


def normalize_entity_id(
    value: Any,
) -> str:
    """Normaliserar ett internt entity-id."""

    if value is None:
        return ""

    return str(value).strip().upper()


class EntityRegistry:
    """Läser och hanterar det persistenta entity-registret."""

    def __init__(
        self,
        *,
        path: Path = ENTITY_REGISTRY_PATH,
    ) -> None:
        self.path = path
        self.records = self._load()
        self._validate()

    # ------------------------------------------------------------------
    # Läsning
    # ------------------------------------------------------------------

    def _load(
        self,
    ) -> list[dict[str, Any]]:
        """Läser entity-registret."""

        if not self.path.exists():
            return []

        records: list[
            dict[str, Any]
        ] = []

        with self.path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            for line_number, line in enumerate(
                handle,
                start=1,
            ):
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(
                        line
                    )

                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "Ogiltig JSON i "
                        f"{self.path} "
                        f"rad {line_number}."
                    ) from exc

                if not isinstance(
                    record,
                    dict,
                ):
                    raise ValueError(
                        "Entity-post på rad "
                        f"{line_number} är inte ett objekt."
                    )

                records.append(
                    record
                )

        return records

    def _validate(
        self,
    ) -> None:
        """Validerar entity-registret."""

        seen_ids: set[str] = set()

        for record in self.records:
            entity_id = normalize_entity_id(
                record.get("entity_id")
            )

            if not entity_id:
                raise ValueError(
                    "Entity-post saknar entity_id."
                )

            if entity_id in seen_ids:
                raise ValueError(
                    "Duplicerat entity_id i "
                    f"{self.path}: {entity_id}"
                )

            if not ENTITY_ID_PATTERN.fullmatch(
                entity_id
            ):
                raise ValueError(
                    "Ogiltigt entity_id i "
                    f"{self.path}: {entity_id}"
                )

            seen_ids.add(
                entity_id
            )

    # ------------------------------------------------------------------
    # Uppslag
    # ------------------------------------------------------------------

    def by_id(
        self,
        entity_id: str,
    ) -> dict[str, Any] | None:
        """Returnerar entity-posten för ett entity-id."""

        normalized = normalize_entity_id(
            entity_id
        )

        for record in self.records:
            if (
                normalize_entity_id(
                    record.get("entity_id")
                )
                == normalized
            ):
                return record

        return None

    def ids(
        self,
    ) -> set[str]:
        """Returnerar alla registrerade entity-id:n."""

        return {
            normalize_entity_id(
                record.get("entity_id")
            )
            for record in self.records
            if record.get("entity_id")
        }

    def contains(
        self,
        entity_id: str,
    ) -> bool:
        """Kontrollerar om entity-id finns."""

        return (
            normalize_entity_id(entity_id)
            in self.ids()
        )

    # ------------------------------------------------------------------
    # ID
    # ------------------------------------------------------------------

    def next_entity_id(
        self,
    ) -> str:
        """Skapar nästa lediga interna entity-id."""

        highest = 0

        for entity_id in self.ids():
            match = ENTITY_ID_PATTERN.fullmatch(
                entity_id
            )

            if not match:
                continue

            highest = max(
                highest,
                int(match.group(1)),
            )

        return (
            f"{ENTITY_ID_PREFIX}"
            f"{highest + 1:0{ENTITY_ID_WIDTH}d}"
        )

    # ------------------------------------------------------------------
    # Entity
    # ------------------------------------------------------------------

    def add(
        self,
        *,
        legal_name: str | None,
        observed_date: str,
        source: str,
        status: str = "active",
        entity_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Lägger till en entity.

        Om entity_id anges används det befintliga interna ID:t.
        Annars skapas nästa lediga ID.
        """

        if entity_id is not None:
            normalized_entity_id = (
                normalize_entity_id(
                    entity_id
                )
            )

            existing = self.by_id(
                normalized_entity_id
            )

            if existing is not None:
                return existing

            if not ENTITY_ID_PATTERN.fullmatch(
                normalized_entity_id
            ):
                raise ValueError(
                    "Ogiltigt entity_id: "
                    f"{entity_id}"
                )

            entity_id = normalized_entity_id

        else:
            entity_id = self.next_entity_id()

        record = {
            "entity_id": entity_id,
            "legal_name": (
                str(legal_name).strip()
                if legal_name is not None
                else None
            ),
            "status": (
                str(status).strip()
                if status
                else "active"
            ),
            "created_date": (
                str(observed_date).strip()
            ),
            "source": (
                str(source).strip()
            ),
        }

        self.records.append(
            record
        )

        return record

    def save(
        self,
    ) -> None:
        """Skriver hela entity-registret."""

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            for record in self.records:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(
                            ",",
                            ":",
                        ),
                    )
                )
                handle.write("\n")
