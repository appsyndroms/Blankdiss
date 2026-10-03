"""Centralt identitetslager för instrument och entiteter.

Modulen håller isär:

    entity
        ↓
    instrument
        ↓
    externa identifierare

En entity representerar den långsiktiga identiteten för en emittent.
Ett instrument representerar ett specifikt värdepapper inom entityn.

Exempel:

    ENT-000001
        ├── SE0007439112
        └── SE0016101844

ISIN:er slås aldrig automatiskt ihop till samma instrument.

Alias-/observationsregistret är append-only och används för att bygga
upp identitetskunskap över tid.

instrument_map.json innehåller befintlig instrumentkunskap men är inte
ett historiskt observationsregister. Träffar därifrån används därför
som instrument-evidens, inte som historiskt giltiga entity-observationer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]

ANALYSIS_DIR = (
    ROOT
    / "data"
    / "analysis"
)

INSTRUMENT_MAP_PATH = (
    ANALYSIS_DIR
    / "instrument_map.json"
)

INSTRUMENT_ALIASES_PATH = (
    ANALYSIS_DIR
    / "instrument_aliases.jsonl"
)


def normalize_text(
    value: Any,
) -> str:
    """Normaliserar identifierande text."""

    if value is None:
        return ""

    text = str(value).strip().upper()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def normalize_isin(
    value: Any,
) -> str:
    """Normaliserar ISIN."""

    return (
        normalize_text(value)
        .replace(" ", "")
    )


def normalize_lei(
    value: Any,
) -> str:
    """Normaliserar LEI."""

    return (
        normalize_text(value)
        .replace(" ", "")
    )


def normalize_identifier(
    value: Any,
) -> str:
    """Normaliserar en generell identifierare."""

    return (
        normalize_text(value)
        .replace(" ", "")
    )


def parse_date(
    value: Any,
) -> date | None:
    """Tolkar ett datum till date."""

    if value is None:
        return None

    if isinstance(
        value,
        date,
    ):
        return value

    text = str(value).strip()

    if not text:
        return None

    try:
        return date.fromisoformat(
            text[:10]
        )

    except ValueError:
        return None


def date_is_valid(
    record: dict[str, Any],
    target_date: date | None,
) -> bool:
    """Kontrollerar om en observation gäller på ett datum."""

    if target_date is None:
        return True

    valid_from = parse_date(
        record.get("valid_from")
    )

    valid_to = parse_date(
        record.get("valid_to")
    )

    if (
        valid_from is not None
        and target_date < valid_from
    ):
        return False

    if (
        valid_to is not None
        and target_date > valid_to
    ):
        return False

    return True


@dataclass(frozen=True)
class IdentityMatch:
    """Resultat från en identitetsmatchning."""

    entity_id: str
    record: dict[str, Any]
    resolution: str


@dataclass(frozen=True)
class InstrumentMatch:
    """Resultat från en instrumentmatchning."""

    isin: str
    record: dict[str, Any]
    resolution: str
    source: str


class InstrumentIdentity:
    """Läser och hanterar Blankdiss identitetsregister."""

    def __init__(
        self,
        *,
        instrument_map_path: Path = INSTRUMENT_MAP_PATH,
        aliases_path: Path = INSTRUMENT_ALIASES_PATH,
    ) -> None:
        self.instrument_map_path = (
            instrument_map_path
        )

        self.aliases_path = (
            aliases_path
        )

        self.instrument_map = (
            self._load_instrument_map()
        )

        self.records = (
            self._load_aliases()
        )

    # ------------------------------------------------------------------
    # Läsning
    # ------------------------------------------------------------------

    def _load_instrument_map(
        self,
    ) -> dict[str, dict[str, Any]]:
        """Läser befintlig instrument_map.json."""

        if not self.instrument_map_path.exists():
            return {}

        content = (
            self.instrument_map_path.read_text(
                encoding="utf-8"
            )
        )

        if not content.strip():
            return {}

        data = json.loads(
            content
        )

        if not isinstance(
            data,
            dict,
        ):
            raise ValueError(
                "instrument_map.json måste "
                "innehålla ett JSON-objekt."
            )

        return {
            str(key): value
            for key, value in data.items()
            if isinstance(
                value,
                dict,
            )
        }

    def _load_aliases(
        self,
    ) -> list[dict[str, Any]]:
        """Läser det append-only identitetsregistret."""

        if not self.aliases_path.exists():
            return []

        records: list[
            dict[str, Any]
        ] = []

        with self.aliases_path.open(
            encoding="utf-8"
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
                        "Ogiltig JSONL i "
                        f"{self.aliases_path} "
                        f"på rad {line_number}."
                    ) from exc

                if not isinstance(
                    record,
                    dict,
                ):
                    raise ValueError(
                        "Identitetsregistret får "
                        "bara innehålla JSON-objekt."
                    )

                records.append(
                    record
                )

        return records

    # ------------------------------------------------------------------
    # Entity
    # ------------------------------------------------------------------

    def _entity_ids(
        self,
    ) -> set[str]:
        """Returnerar befintliga entity-id:n."""

        return {
            str(record["entity_id"])
            for record in self.records
            if record.get("entity_id")
        }

    def _next_entity_id(
        self,
    ) -> str:
        """Skapar nästa sekventiella entity-id."""

        numbers: list[int] = []

        for entity_id in self._entity_ids():
            match = re.fullmatch(
                r"ENT-(\d+)",
                entity_id,
            )

            if match:
                numbers.append(
                    int(match.group(1)
                    )
                )

        next_number = (
            max(numbers, default=0)
            + 1
        )

        return (
            f"ENT-{next_number:06d}"
        )

    # ------------------------------------------------------------------
    # Alias-matchning
    # ------------------------------------------------------------------

    def _matching_records(
        self,
        *,
        isin: str | None = None,
        lei: str | None = None,
        issuer: str | None = None,
        ticker: str | None = None,
        yahoo_symbol: str | None = None,
        target_date: date | None = None,
    ) -> list[IdentityMatch]:
        """Hittar identitetsobservationer."""

        normalized_isin = normalize_isin(
            isin
        )

        normalized_lei = normalize_lei(
            lei
        )

        normalized_issuer = normalize_text(
            issuer
        )

        normalized_ticker = normalize_identifier(
            ticker
        )

        normalized_yahoo = normalize_identifier(
            yahoo_symbol
        )

        matches: list[
            IdentityMatch
        ] = []

        for record in self.records:
            if not date_is_valid(
                record,
                target_date,
            ):
                continue

            record_isin = normalize_isin(
                record.get("isin")
            )

            record_lei = normalize_lei(
                record.get("lei")
            )

            record_issuer = normalize_text(
                record.get("issuer")
            )

            record_ticker = normalize_identifier(
                record.get("ticker")
            )

            record_yahoo = normalize_identifier(
                record.get("yahoo_symbol")
            )

            resolution: str | None = None

            if (
                normalized_isin
                and record_isin
                and normalized_isin
                == record_isin
            ):
                resolution = "isin"

            elif (
                normalized_lei
                and record_lei
                and normalized_lei
                == record_lei
            ):
                resolution = "lei"

            elif (
                normalized_yahoo
                and record_yahoo
                and normalized_yahoo
                == record_yahoo
            ):
                resolution = "yahoo_symbol"

            elif (
                normalized_ticker
                and record_ticker
                and normalized_ticker
                == record_ticker
            ):
                resolution = "ticker"

            elif (
                normalized_issuer
                and record_issuer
                and normalized_issuer
                == record_issuer
            ):
                resolution = "issuer"

            if resolution is not None:
                matches.append(
                    IdentityMatch(
                        entity_id=str(
                            record[
                                "entity_id"
                            ]
                        ),
                        record=record,
                        resolution=resolution,
                    )
                )

        return matches

    # ------------------------------------------------------------------
    # Instrument-matchning
    # ------------------------------------------------------------------

    def _matching_instrument_map(
        self,
        *,
        isin: str | None = None,
        lei: str | None = None,
        issuer: str | None = None,
        ticker: str | None = None,
        yahoo_symbol: str | None = None,
    ) -> list[InstrumentMatch]:
        """
        Hittar instrument i instrument_map.json.

        instrument_map är inte tidsstämplad. Träffarna representerar
        därför känd instrumentidentitet, men inte historisk giltighet
        på ett specifikt datum.
        """

        normalized_isin = normalize_isin(
            isin
        )

        normalized_lei = normalize_lei(
            lei
        )

        normalized_issuer = normalize_text(
            issuer
        )

        normalized_ticker = normalize_identifier(
            ticker
        )

        normalized_yahoo = normalize_identifier(
            yahoo_symbol
        )

        matches: list[
            InstrumentMatch
        ] = []

        priority = {
            "isin": 0,
            "lei": 1,
            "yahoo_symbol": 2,
            "ticker": 3,
            "issuer": 4,
        }

        for map_key, record in self.instrument_map.items():
            record_isin = normalize_isin(
                record.get("isin")
            )

            record_lei = normalize_lei(
                record.get("lei")
            )

            record_issuer = normalize_text(
                record.get("issuer")
            )

            record_ticker = normalize_identifier(
                record.get("ticker")
            )

            record_yahoo = normalize_identifier(
                record.get("yahoo_symbol")
            )

            resolution: str | None = None

            if (
                normalized_isin
                and record_isin
                and normalized_isin
                == record_isin
            ):
                resolution = "isin"

            elif (
                normalized_lei
                and record_lei
                and normalized_lei
                == record_lei
            ):
                resolution = "lei"

            elif (
                normalized_yahoo
                and record_yahoo
                and normalized_yahoo
                == record_yahoo
            ):
                resolution = "yahoo_symbol"

            elif (
                normalized_ticker
                and record_ticker
                and normalized_ticker
                == record_ticker
            ):
                resolution = "ticker"

            elif (
                normalized_issuer
                and record_issuer
                and normalized_issuer
                == record_issuer
            ):
                resolution = "issuer"

            if resolution is None:
                continue

            instrument_isin = (
                record_isin
            )

            if not instrument_isin:
                continue

            matches.append(
                InstrumentMatch(
                    isin=instrument_isin,
                    record={
                        **record,
                        "map_key": map_key,
                    },
                    resolution=resolution,
                    source="instrument_map",
                )
            )

        matches.sort(
            key=lambda match: (
                priority[
                    match.resolution
                ],
                match.isin,
            )
        )

        return matches

    def _matching_alias_instruments(
        self,
        *,
        isin: str | None = None,
        lei: str | None = None,
        issuer: str | None = None,
        ticker: str | None = None,
        yahoo_symbol: str | None = None,
        target_date: date | None = None,
    ) -> list[InstrumentMatch]:
        """
        Hittar instrument från det historiska aliasregistret.

        Endast observationer som gäller på target_date tas med.
        """

        matches = self._matching_records(
            isin=isin,
            lei=lei,
            issuer=issuer,
            ticker=ticker,
            yahoo_symbol=yahoo_symbol,
            target_date=target_date,
        )

        result: list[
            InstrumentMatch
        ] = []

        for match in matches:
            instrument_isin = normalize_isin(
                match.record.get("isin")
            )

            if not instrument_isin:
                continue

            result.append(
                InstrumentMatch(
                    isin=instrument_isin,
                    record=match.record,
                    resolution=match.resolution,
                    source="aliases",
                )
            )

        return result

    def _unique_instrument_matches(
        self,
        matches: Iterable[InstrumentMatch],
    ) -> list[InstrumentMatch]:
        """Tar bort dubbletter per ISIN."""

        result: dict[
            str,
            InstrumentMatch,
        ] = {}

        for match in matches:
            result.setdefault(
                match.isin,
                match,
            )

        return list(
            result.values()
        )

    def resolve_instrument(
        self,
        *,
        isin: str | None = None,
        lei: str | None = None,
        issuer: str | None = None,
        ticker: str | None = None,
        yahoo_symbol: str | None = None,
        target_date: date | str | None = None,
    ) -> list[InstrumentMatch]:
        """
        Returnerar kända instrument som matchar identifierarna.

        Resultatet kan innehålla flera ISIN eftersom en entity kan ha
        flera instrument eller eftersom identifieraren kan vara historiskt
        tvetydig.

        Aliasregistret är tidsbegränsat.
        instrument_map är inte tidsbegränsad.
        """

        parsed_date = parse_date(
            target_date
        )

        alias_matches = (
            self._matching_alias_instruments(
                isin=isin,
                lei=lei,
                issuer=issuer,
                ticker=ticker,
                yahoo_symbol=yahoo_symbol,
                target_date=parsed_date,
            )
        )

        map_matches = (
            self._matching_instrument_map(
                isin=isin,
                lei=lei,
                issuer=issuer,
                ticker=ticker,
                yahoo_symbol=yahoo_symbol,
            )
        )

        combined = (
            alias_matches
            + map_matches
        )

        return self._unique_instrument_matches(
            combined
        )

    # ------------------------------------------------------------------
    # Publika entity-resolve-metoder
    # ------------------------------------------------------------------

    def resolve(
        self,
        *,
        entity_id: str | None = None,
        isin: str | None = None,
        lei: str | None = None,
        issuer: str | None = None,
        ticker: str | None = None,
        yahoo_symbol: str | None = None,
        target_date: date | str | None = None,
    ) -> IdentityMatch | None:
        """
        Löser en entity.

        Prioritet:

            entity_id
            ↓
            ISIN
            ↓
            LEI
            ↓
            Yahoo-symbol
            ↓
            ticker
            ↓
            issuer

        instrument_map används inte för att skapa en entity-match,
        eftersom instrument_map saknar entity_id.

        Om flera entitys matchar samma identifierare returneras
        ingen godtycklig träff.
        """

        parsed_date = parse_date(
            target_date
        )

        if entity_id:
            normalized_entity = (
                normalize_identifier(
                    entity_id
                )
            )

            matches = [
                record
                for record in self.records
                if normalize_identifier(
                    record.get(
                        "entity_id"
                    )
                )
                == normalized_entity
                and date_is_valid(
                    record,
                    parsed_date,
                )
            ]

            if not matches:
                return None

            return IdentityMatch(
                entity_id=normalized_entity,
                record=matches[0],
                resolution="entity_id",
            )

        candidates = self._matching_records(
            isin=isin,
            lei=lei,
            issuer=issuer,
            ticker=ticker,
            yahoo_symbol=yahoo_symbol,
            target_date=parsed_date,
        )

        if not candidates:
            return None

        # Identifierare med högre precision vinner.
        priority = {
            "isin": 0,
            "lei": 1,
            "yahoo_symbol": 2,
            "ticker": 3,
            "issuer": 4,
        }

        candidates.sort(
            key=lambda match: priority[
                match.resolution
            ]
        )

        best_priority = priority[
            candidates[0].resolution
        ]

        best = [
            match
            for match in candidates
            if priority[
                match.resolution
            ]
            == best_priority
        ]

        unique = self._unique_entities(
            best
        )

        if len(unique) != 1:
            return None

        return unique[0]

    def resolve_entity(
        self,
        **kwargs: Any,
    ) -> str | None:
        """Returnerar endast entity_id."""

        match = self.resolve(
            **kwargs
        )

        if match is None:
            return None

        return match.entity_id

    # ------------------------------------------------------------------
    # Entity → instrument
    # ------------------------------------------------------------------

    def instruments_for_entity(
        self,
        entity_id: str,
        target_date: date | str | None = None,
    ) -> list[dict[str, Any]]:
        """Returnerar alla kända instrument för en entity."""

        parsed_date = parse_date(
            target_date
        )

        normalized_entity = (
            normalize_identifier(
                entity_id
            )
        )

        result: list[
            dict[str, Any]
        ] = []

        seen: set[str] = set()

        for record in self.records:
            if (
                normalize_identifier(
                    record.get(
                        "entity_id"
                    )
                )
                != normalized_entity
            ):
                continue

            if not date_is_valid(
                record,
                parsed_date,
            ):
                continue

            isin = normalize_isin(
                record.get("isin")
            )

            key = isin or (
                "issuer:"
                + normalize_text(
                    record.get(
                        "issuer"
                    )
                )
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(
                record
            )

        return result

    def instrument_for_entity(
        self,
        entity_id: str,
        target_date: date | str,
    ) -> dict[str, Any] | None:
        """
        Returnerar instrumentet som gäller för en entity på ett datum.

        Om flera kandidater gäller samtidigt returneras ingen godtycklig
        kandidat.
        """

        instruments = (
            self.instruments_for_entity(
                entity_id,
                target_date,
            )
        )

        if not instruments:
            return None

        if len(instruments) == 1:
            return instruments[0]

        return None

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover(
        self,
        *,
        isin: str | None,
        issuer: str | None,
        lei: str | None,
        source: str,
        observed_date: date | str | None = None,
        valid_from: date | str | None = None,
        valid_to: date | str | None = None,
        relation: str = "observed",
        status: str = "candidate",
        ticker: str | None = None,
        yahoo_symbol: str | None = None,
    ) -> dict[str, Any]:
        """
        Registrerar en ny identitetsobservation.

        Discovery är medvetet försiktig:

        - befintligt ISIN återanvänder entity
        - entydigt LEI återanvänder entity
        - entydig issuer kan återanvända entity
        - annars skapas en ny entity

        Observationen appendas bara om exakt samma observation inte
        redan finns.
        """

        normalized_isin = normalize_isin(
            isin
        )

        normalized_lei = normalize_lei(
            lei
        )

        normalized_issuer = normalize_text(
            issuer
        )

        if not (
            normalized_isin
            or normalized_lei
            or normalized_issuer
        ):
            raise ValueError(
                "Discovery kräver minst "
                "isin, lei eller issuer."
            )

        existing = self._matching_records(
            isin=normalized_isin or None,
            lei=normalized_lei or None,
            issuer=normalized_issuer or None,
        )

        # ISIN är starkast. Om det redan finns väljer vi den entityn.
        isin_matches = [
            match
            for match in existing
            if match.resolution == "isin"
        ]

        if len(
            self._unique_entities(
                isin_matches
            )
        ) == 1:
            entity_id = (
                isin_matches[0].entity_id
            )

        else:
            lei_matches = [
                match
                for match in existing
                if match.resolution == "lei"
            ]

            unique_lei = (
                self._unique_entities(
                    lei_matches
                )
            )

            if len(unique_lei) == 1:
                entity_id = (
                    unique_lei[0].entity_id
                )

            else:
                issuer_matches = [
                    match
                    for match in existing
                    if match.resolution
                    == "issuer"
                ]

                unique_issuer = (
                    self._unique_entities(
                        issuer_matches
                    )
                )

                if len(unique_issuer) == 1:
                    entity_id = (
                        unique_issuer[
                            0
                        ].entity_id
                    )

                else:
                    entity_id = (
                        self._next_entity_id()
                    )

        record = {
            "entity_id": entity_id,
            "isin": (
                normalized_isin
                or None
            ),
            "issuer": (
                str(issuer).strip()
                if issuer is not None
                else None
            ),
            "lei": (
                normalized_lei
                or None
            ),
            "valid_from": (
                parse_date(
                    valid_from
                ).isoformat()
                if parse_date(
                    valid_from
                )
                else None
            ),
            "valid_to": (
                parse_date(
                    valid_to
                ).isoformat()
                if parse_date(
                    valid_to
                )
                else None
            ),
            "relation": relation,
            "source": str(
                source
            ).strip(),
            "status": status,
        }

        if ticker:
            record["ticker"] = (
                str(ticker).strip()
            )

        if yahoo_symbol:
            record["yahoo_symbol"] = (
                str(yahoo_symbol).strip()
            )

        if observed_date is not None:
            parsed_observed = parse_date(
                observed_date
            )

            if parsed_observed is None:
                raise ValueError(
                    "observed_date har "
                    "ogiltigt datum: "
                    f"{observed_date!r}"
                )

            record[
                "observed_date"
            ] = parsed_observed.isoformat()

        if not self._observation_exists(
            record
        ):
            self._append_record(
                record
            )
            self.records.append(
                record
            )

        return record

    def _observation_exists(
        self,
        record: dict[str, Any],
    ) -> bool:
        """Kontrollerar om exakt observation redan finns."""

        keys = (
            "entity_id",
            "isin",
            "issuer",
            "lei",
            "valid_from",
            "valid_to",
            "relation",
            "source",
            "status",
            "observed_date",
        )

        for existing in self.records:
            if all(
                existing.get(key)
                == record.get(key)
                for key in keys
            ):
                return True

        return False

    def _append_record(
        self,
        record: dict[str, Any],
    ) -> None:
        """Append-only skrivning till JSONL."""

        self.aliases_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.aliases_path.open(
            "a",
            encoding="utf-8"
        ) as handle:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(
                        ",",
                        ":",
                    ),
                )
                + "\n"
            )


def load_identity() -> InstrumentIdentity:
    """Skapar standardidentitetslagret."""

    return InstrumentIdentity()
