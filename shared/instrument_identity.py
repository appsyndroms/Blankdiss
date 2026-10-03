"""Centralt identitetslager för instrument och entities.

Identity-lagret håller isär:

    entity
        ↓
    instrument
        ↓
    externa identifierare

En entity representerar den långsiktiga identiteten för en
juridisk/emittentmässig enhet.

Ett instrument representerar ett specifikt värdepapper.

Exempel:

    ENT-000001
        ├── ISIN A
        ├── ISIN B
        ├── LEI A
        └── LEI B

ISIN är instrumentidentitet.
LEI, ticker och Yahoo-symboler är externa identifierare/alias.

instrument_entities.jsonl innehåller de interna, persistenta
entity-identiteterna.

instrument_aliases.jsonl innehåller historiska observationer
och relationer mellan entities, instrument och externa
identifierare.

instrument_map.json är däremot Blankdiss-specifik instrumentdata
och ligger kvar under data/analysis.

Identity contract:

    ISIN → exakt en entity
    LEI  → exakt en entity

Ett byte av LEI är därför tillåtet när den nya observationen
fortfarande pekar på samma entity.

Däremot får samma ISIN eller LEI aldrig börja peka på en annan
entity utan att detta uttryckligen hanteras som en konflikt.

Yahoo-symbol, ticker och issuer-namn är observerade attribut
och får förändras över tid.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from shared.entity_identity import (
    ENTITY_REGISTRY_PATH,
    EntityRegistry,
    SHARED_DATA_DIR,
)


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
    SHARED_DATA_DIR
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
    """Normaliserar generell identifierare."""
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


@dataclass(frozen=True)
class IdentityConflict:
    """Konflikt mot identity contract."""

    identifier_type: str
    identifier: str
    existing_entity_ids: tuple[str, ...]
    proposed_entity_id: str | None
    message: str


class IdentityContractError(ValueError):
    """Raised when an identity observation violates the contract."""


class InstrumentIdentity:
    """Centralt identitetslager för instrument."""

    def __init__(
        self,
        *,
        instrument_map_path: Path = INSTRUMENT_MAP_PATH,
        aliases_path: Path = INSTRUMENT_ALIASES_PATH,
        entity_registry_path: Path = ENTITY_REGISTRY_PATH,
    ) -> None:
        self.instrument_map_path = (
            instrument_map_path
        )

        self.aliases_path = (
            aliases_path
        )

        self.entity_registry = (
            EntityRegistry(
                path=entity_registry_path
            )
        )

        self.instrument_map = (
            self._load_instrument_map()
        )

        self.records = (
            self._load_aliases()
        )

        self._instrument_map_indexes = (
            self._build_instrument_map_indexes()
        )

        self._alias_instrument_indexes = (
            self._build_alias_instrument_indexes()
        )

        self.validate_identity_data()

    def _load_instrument_map(
        self,
    ) -> dict[str, dict[str, Any]]:
        """Läser Blankdiss instrument_map.json."""

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
        """Läser shared/data/instrument_aliases.jsonl."""

        if not self.aliases_path.exists():
            return []

        records: list[
            dict[str, Any]
        ] = []

        with self.aliases_path.open(
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
                        "Ogiltig JSONL i "
                        f"{self.aliases_path} "
                        f"på rad {line_number}."
                    ) from exc

                if not isinstance(
                    record,
                    dict,
                ):
                    raise ValueError(
                        "Aliaspost på rad "
                        f"{line_number} är inte ett objekt."
                    )

                records.append(
                    record
                )

        return records

    def _build_instrument_map_indexes(
        self,
    ) -> dict[str, dict[str, list[str]]]:
        """Bygger index för instrument_map."""

        indexes: dict[
            str,
            dict[str, list[str]],
        ] = {
            "isin": {},
            "lei": {},
            "issuer": {},
            "ticker": {},
            "yahoo_symbol": {},
        }

        fields = (
            "isin",
            "lei",
            "issuer",
            "ticker",
            "yahoo_symbol",
        )

        normalizers = {
            "isin": normalize_isin,
            "lei": normalize_lei,
            "issuer": normalize_text,
            "ticker": normalize_identifier,
            "yahoo_symbol": normalize_identifier,
        }

        for map_key, record in self.instrument_map.items():
            for field in fields:
                normalized = normalizers[field](
                    record.get(field)
                )

                if not normalized:
                    continue

                indexes[field].setdefault(
                    normalized,
                    [],
                ).append(
                    map_key
                )

        return indexes

    def _build_alias_instrument_indexes(
        self,
    ) -> dict[str, list[int]]:
        """Bygger index för aliasregistret."""

        indexes: dict[
            str,
            list[int],
        ] = {}

        for index, record in enumerate(
            self.records
        ):
            isin = normalize_isin(
                record.get("isin")
            )

            if not isin:
                continue

            indexes.setdefault(
                isin,
                [],
            ).append(
                index
            )

        return indexes

    def _entity_ids(
        self,
    ) -> set[str]:
        """Returnerar entity-id:n från centrala registret."""

        return self.entity_registry.ids()

    def _next_entity_id(
        self,
    ) -> str:
        """Returnerar nästa ID från centrala registret."""

        return (
            self.entity_registry.next_entity_id()
        )

    def _register_entity(
        self,
        *,
        entity_id: str,
        legal_name: str | None,
        observed_date: str | None,
        source: str,
        status: str,
    ) -> None:
        """Säkerställer att entity finns i registret."""

        if self.entity_registry.contains(
            entity_id
        ):
            return

        if not observed_date:
            observed_date = (
                date.today().isoformat()
            )

        self.entity_registry.add(
            entity_id=entity_id,
            legal_name=legal_name,
            observed_date=observed_date,
            source=source,
            status=status,
        )

        self.entity_registry.save()

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
                and normalized_isin == record_isin
            ):
                resolution = "isin"

            elif (
                normalized_lei
                and record_lei
                and normalized_lei == record_lei
            ):
                resolution = "lei"

            elif (
                normalized_yahoo
                and record_yahoo
                and normalized_yahoo == record_yahoo
            ):
                resolution = "yahoo_symbol"

            elif (
                normalized_ticker
                and record_ticker
                and normalized_ticker == record_ticker
            ):
                resolution = "ticker"

            elif (
                normalized_issuer
                and record_issuer
                and normalized_issuer == record_issuer
            ):
                resolution = "issuer"

            if resolution is not None:
                matches.append(
                    IdentityMatch(
                        entity_id=str(
                            record["entity_id"]
                        ),
                        record=record,
                        resolution=resolution,
                    )
                )

        return matches

    def _matching_instrument_map(
        self,
        *,
        isin: str | None = None,
        lei: str | None = None,
        issuer: str | None = None,
        ticker: str | None = None,
        yahoo_symbol: str | None = None,
    ) -> list[InstrumentMatch]:
        """Hittar instrument i instrument_map."""

        normalized_values = {
            "isin": normalize_isin(isin),
            "lei": normalize_lei(lei),
            "issuer": normalize_text(issuer),
            "ticker": normalize_identifier(ticker),
            "yahoo_symbol": normalize_identifier(
                yahoo_symbol
            ),
        }

        priority = {
            "isin": 0,
            "lei": 1,
            "yahoo_symbol": 2,
            "ticker": 3,
            "issuer": 4,
        }

        candidate_resolutions: dict[
            str,
            str,
        ] = {}

        for resolution in (
            "isin",
            "lei",
            "yahoo_symbol",
            "ticker",
            "issuer",
        ):
            normalized = normalized_values[
                resolution
            ]

            if not normalized:
                continue

            for map_key in (
                self._instrument_map_indexes[
                    resolution
                ].get(
                    normalized,
                    [],
                )
            ):
                existing = candidate_resolutions.get(
                    map_key
                )

                if (
                    existing is None
                    or priority[resolution]
                    < priority[existing]
                ):
                    candidate_resolutions[
                        map_key
                    ] = resolution

        matches: list[
            InstrumentMatch
        ] = []

        for map_key, resolution in (
            candidate_resolutions.items()
        ):
            record = self.instrument_map.get(
                map_key
            )

            if record is None:
                continue

            instrument_isin = normalize_isin(
                record.get("isin")
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
                priority[match.resolution],
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
        """Hittar instrument från aliasregistret."""

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

    def _identifier_entities(
        self,
        *,
        field: str,
    ) -> dict[str, set[str]]:
        """Bygger identifierare -> entities."""

        result: dict[
            str,
            set[str],
        ] = {}

        normalizer = (
            normalize_isin
            if field == "isin"
            else normalize_lei
        )

        for record in self.records:
            value = normalizer(
                record.get(field)
            )

            entity_id = normalize_identifier(
                record.get("entity_id")
            )

            if not value or not entity_id:
                continue

            result.setdefault(
                value,
                set(),
            ).add(
                entity_id
            )

        return result

    def validate_identity_data(
        self,
    ) -> None:
        """
        Validerar hela identity-registret mot identity contract.

        Regler:

            1. Alla alias måste referera till en befintlig entity.
            2. Ett ISIN får endast tillhöra en entity.
            3. Ett LEI får endast tillhöra en entity.

        Yahoo-symbol, ticker och issuer får däremot förekomma
        på flera observationer eftersom de kan förändras eller
        återanvändas över tid.
        """

        entity_ids = self._entity_ids()

        for index, record in enumerate(
            self.records,
            start=1,
        ):
            entity_id = normalize_identifier(
                record.get("entity_id")
            )

            if not entity_id:
                raise IdentityContractError(
                    "Aliaspost saknar entity_id "
                    f"(rad {index})."
                )

            if entity_id not in entity_ids:
                raise IdentityContractError(
                    "Aliaspost refererar till okänd "
                    f"entity_id {entity_id} "
                    f"(rad {index})."
                )

        for field in (
            "isin",
            "lei",
        ):
            mapping = self._identifier_entities(
                field=field
            )

            for identifier, mapped_entities in (
                mapping.items()
            ):
                if len(mapped_entities) <= 1:
                    continue

                raise IdentityContractError(
                    "Identity-konflikt: "
                    f"{field.upper()} {identifier} "
                    "är kopplad till flera entities: "
                    f"{', '.join(sorted(mapped_entities))}."
                )

    def validate_observation(
        self,
        record: dict[str, Any],
        *,
        existing_records: Iterable[
            dict[str, Any]
        ] | None = None,
    ) -> list[IdentityConflict]:
        """
        Validerar en ny observation mot identity contract.

        Tillåtet:

            samma ISIN + ny LEI + samma entity
            samma ISIN + nytt issuer-namn + samma entity
            samma ISIN + ny Yahoo-symbol + samma entity

        Konflikt:

            samma ISIN + annan entity
            samma LEI + annan entity

        Resultatet är en lista eftersom en observation kan
        bryta mot flera regler samtidigt.
        """

        records = (
            list(existing_records)
            if existing_records is not None
            else self.records
        )

        proposed_entity_id = normalize_identifier(
            record.get("entity_id")
        )

        isin = normalize_isin(
            record.get("isin")
        )

        lei = normalize_lei(
            record.get("lei")
        )

        conflicts: list[
            IdentityConflict
        ] = []

        if proposed_entity_id:
            entity_ids = self._entity_ids()

            if proposed_entity_id not in entity_ids:
                conflicts.append(
                    IdentityConflict(
                        identifier_type="entity_id",
                        identifier=proposed_entity_id,
                        existing_entity_ids=(),
                        proposed_entity_id=proposed_entity_id,
                        message=(
                            "Observationen refererar till "
                            "en entity som inte finns."
                        ),
                    )
                )

        for field, value in (
            ("isin", isin),
            ("lei", lei),
        ):
            if not value:
                continue

            normalizer = (
                normalize_isin
                if field == "isin"
                else normalize_lei
            )

            mapped_entities: set[str] = set()

            for existing in records:
                existing_value = normalizer(
                    existing.get(field)
                )

                if existing_value != value:
                    continue

                existing_entity = normalize_identifier(
                    existing.get("entity_id")
                )

                if existing_entity:
                    mapped_entities.add(
                        existing_entity
                    )

            foreign_entities = (
                mapped_entities
                - {proposed_entity_id}
            )

            if not foreign_entities:
                continue

            conflicts.append(
                IdentityConflict(
                    identifier_type=field,
                    identifier=value,
                    existing_entity_ids=tuple(
                        sorted(foreign_entities)
                    ),
                    proposed_entity_id=(
                        proposed_entity_id
                        or None
                    ),
                    message=(
                        f"{field.upper()} {value} är redan "
                        "kopplad till annan entity: "
                        + ", ".join(
                            sorted(
                                foreign_entities
                            )
                        )
                        + "."
                    ),
                )
            )

        return conflicts

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
        """Returnerar kända instrument."""

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

        return self._unique_instrument_matches(
            alias_matches + map_matches
        )

    def entities_for_instrument(
        self,
        isin: str,
        target_date: date | str | None = None,
    ) -> list[IdentityMatch]:
        """Returnerar entities för ett instrument."""

        normalized_isin = normalize_isin(
            isin
        )

        if not normalized_isin:
            return []

        parsed_date = parse_date(
            target_date
        )

        result: list[
            IdentityMatch
        ] = []

        record_indexes = (
            self._alias_instrument_indexes.get(
                normalized_isin,
                [],
            )
        )

        for record_index in record_indexes:
            record = self.records[
                record_index
            ]

            if not date_is_valid(
                record,
                parsed_date,
            ):
                continue

            entity_id = record.get(
                "entity_id"
            )

            if not entity_id:
                continue

            result.append(
                IdentityMatch(
                    entity_id=str(
                        entity_id
                    ),
                    record=record,
                    resolution="isin",
                )
            )

        return result

    def unique_entities_for_instrument(
        self,
        isin: str,
        target_date: date | str | None = None,
    ) -> list[str]:
        """Returnerar unika entity-id:n."""

        result: list[str] = []

        seen: set[str] = set()

        for match in self.entities_for_instrument(
            isin,
            target_date,
        ):
            if match.entity_id in seen:
                continue

            seen.add(
                match.entity_id
            )

            result.append(
                match.entity_id
            )

        return result

    def entity_for_instrument(
        self,
        isin: str,
        target_date: date | str | None = None,
    ) -> IdentityMatch | None:
        """Returnerar entity om instrumentet är entydigt."""

        matches = (
            self.entities_for_instrument(
                isin,
                target_date,
            )
        )

        unique: dict[
            str,
            IdentityMatch,
        ] = {}

        for match in matches:
            unique.setdefault(
                match.entity_id,
                match,
            )

        if len(unique) != 1:
            return None

        return next(
            iter(unique.values())
        )

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
        """Löser en entity."""

        parsed_date = parse_date(
            target_date
        )

        if entity_id:
            normalized_entity = normalize_identifier(
                entity_id
            )

            matches = [
                record
                for record in self.records
                if normalize_identifier(
                    record.get("entity_id")
                ) == normalized_entity
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

        priority = {
            "isin": 0,
            "lei": 1,
            "yahoo_symbol": 2,
            "ticker": 3,
            "issuer": 4,
        }

        best_priority = min(
            priority[
                match.resolution
            ]
            for match in candidates
        )

        best = [
            match
            for match in candidates
            if priority[
                match.resolution
            ] == best_priority
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
        """Returnerar endast entity-id."""

        match = self.resolve(
            **kwargs
        )

        return (
            match.entity_id
            if match is not None
            else None
        )

    def instruments_for_entity(
        self,
        entity_id: str,
        target_date: date | str | None = None,
    ) -> list[dict[str, Any]]:
        """Returnerar alla kända instrument för entity."""

        parsed_date = parse_date(
            target_date
        )

        normalized_entity = normalize_identifier(
            entity_id
        )

        result: list[
            dict[str, Any]
        ] = []

        seen: set[str] = set()

        for record in self.records:
            if normalize_identifier(
                record.get("entity_id")
            ) != normalized_entity:
                continue

            if not date_is_valid(
                record,
                parsed_date,
            ):
                continue

            isin = normalize_isin(
                record.get("isin")
            )

            key = (
                isin
                or "issuer:"
                + normalize_text(
                    record.get("issuer")
                )
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            result.append(
                record
            )

        return result

    def instrument_for_entity(
        self,
        entity_id: str,
        target_date: date | str,
    ) -> dict[str, Any] | None:
        """Returnerar instrument om exakt ett är giltigt."""

        instruments = (
            self.instruments_for_entity(
                entity_id,
                target_date,
            )
        )

        if len(instruments) != 1:
            return None

        return instruments[0]

    def _unique_entities(
        self,
        matches: Iterable[IdentityMatch],
    ) -> list[IdentityMatch]:
        """Tar bort dubbletter per entity-id."""

        result: dict[
            str,
            IdentityMatch,
        ] = {}

        for match in matches:
            result.setdefault(
                match.entity_id,
                match
            )

        return list(
            result.values()
        )

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
        Registrerar en identitetsobservation.

        Matchningsprioritet:

            ISIN
            ↓
            LEI
            ↓
            issuer
            ↓
            ny persistent entity

        Ett nytt entity-id skapas endast i det centrala
        EntityRegistry.

        Identity contract valideras innan observationen skrivs.
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

        isin_matches = [
            match
            for match in existing
            if match.resolution == "isin"
        ]

        unique_isin = self._unique_entities(
            isin_matches
        )

        if len(unique_isin) == 1:
            entity_id = unique_isin[0].entity_id

        elif len(unique_isin) > 1:
            raise IdentityContractError(
                "Identity-konflikt: ISIN "
                f"{normalized_isin} är kopplad till "
                "flera entities: "
                + ", ".join(
                    sorted(
                        match.entity_id
                        for match in unique_isin
                    )
                )
                + "."
            )

        else:
            lei_matches = [
                match
                for match in existing
                if match.resolution == "lei"
            ]

            unique_lei = self._unique_entities(
                lei_matches
            )

            if len(unique_lei) == 1:
                entity_id = unique_lei[0].entity_id

            elif len(unique_lei) > 1:
                raise IdentityContractError(
                    "Identity-konflikt: LEI "
                    f"{normalized_lei} är kopplad till "
                    "flera entities: "
                    + ", ".join(
                        sorted(
                            match.entity_id
                            for match in unique_lei
                        )
                    )
                    + "."
                )

            else:
                issuer_matches = [
                    match
                    for match in existing
                    if match.resolution == "issuer"
                ]

                unique_issuer = self._unique_entities(
                    issuer_matches
                )

                if len(unique_issuer) == 1:
                    entity_id = (
                        unique_issuer[0].entity_id
                    )

                else:
                    entity_id = (
                        self._next_entity_id()
                    )

        parsed_observed = parse_date(
            observed_date
        )

        observed_text = (
            parsed_observed.isoformat()
            if parsed_observed
            else None
        )

        parsed_valid_from = parse_date(
            valid_from
        )

        parsed_valid_to = parse_date(
            valid_to
        )

        record = {
            "entity_id": entity_id,
            "isin": (
                normalized_isin
                or None
            ),
            "issuer": (
                normalized_issuer
                or None
            ),
            "lei": (
                normalized_lei
                or None
            ),
            "valid_from": (
                parsed_valid_from.isoformat()
                if parsed_valid_from
                else None
            ),
            "valid_to": (
                parsed_valid_to.isoformat()
                if parsed_valid_to
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

        if observed_text:
            record["observed_date"] = (
                observed_text
            )

        conflicts = self.validate_observation(
            record
        )

        if conflicts:
            raise IdentityContractError(
                "\n".join(
                    conflict.message
                    for conflict in conflicts
                )
            )

        self._register_entity(
            entity_id=entity_id,
            legal_name=(
                str(issuer).strip()
                if issuer is not None
                else None
            ),
            observed_date=observed_text,
            source=source,
            status=status,
        )

        if not self._observation_exists(
            record
        ):
            self._append_record(
                record
            )

            self.records.append(
                record
            )

            record_isin = normalize_isin(
                record.get("isin")
            )

            if record_isin:
                self._alias_instrument_indexes.setdefault(
                    record_isin,
                    [],
                ).append(
                    len(self.records) - 1
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

        return any(
            all(
                existing.get(key)
                == record.get(key)
                for key in keys
            )
            for existing in self.records
        )

    def _append_record(
        self,
        record: dict[str, Any],
    ) -> None:
        """Append-only skrivning till aliasregistret."""

        self.aliases_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.aliases_path.open(
            "a",
            encoding="utf-8",
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
