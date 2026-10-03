from pathlib import Path

import pytest

from shared.instrument_identity import (
    IdentityContractError,
    InstrumentIdentity,
)


def _identity(tmp_path: Path) -> InstrumentIdentity:
    return InstrumentIdentity(
        instrument_map_path=tmp_path / "instrument_map.json",
        aliases_path=tmp_path / "instrument_aliases.jsonl",
        entity_registry_path=tmp_path / "instrument_entities.jsonl",
    )


def _records(identity: InstrumentIdentity) -> list[dict]:
    return identity.records


def test_same_isin_with_new_name_keeps_same_entity(
    tmp_path,
):
    identity = _identity(tmp_path)

    first = identity.discover(
        isin="SE0000000001",
        issuer="Example AB",
        lei="LEI000000000000000001",
        source="test",
        observed_date="2026-01-01",
        status="observed",
    )

    second = identity.discover(
        isin="SE0000000001",
        issuer="Example New Name AB",
        lei="LEI000000000000000001",
        source="test",
        observed_date="2026-02-01",
        status="observed",
    )

    assert first["entity_id"] == second["entity_id"]
    assert len(_records(identity)) == 2

    assert _records(identity)[0]["issuer"] == "EXAMPLE AB"
    assert _records(identity)[1]["issuer"] == "EXAMPLE NEW NAME AB"


def test_same_isin_with_new_yahoo_symbol_keeps_same_entity_and_new_observation(
    tmp_path,
):
    identity = _identity(tmp_path)

    first = identity.discover(
        isin="SE0000000002",
        issuer="Example AB",
        lei="LEI000000000000000002",
        yahoo_symbol="OLD.ST",
        source="test",
        observed_date="2026-01-01",
        status="observed",
    )

    second = identity.discover(
        isin="SE0000000002",
        issuer="Example AB",
        lei="LEI000000000000000002",
        yahoo_symbol="NEW.ST",
        source="test",
        observed_date="2026-02-01",
        status="observed",
    )

    assert first["entity_id"] == second["entity_id"]
    assert len(_records(identity)) == 2

    assert _records(identity)[0]["yahoo_symbol"] == "OLD.ST"
    assert _records(identity)[1]["yahoo_symbol"] == "NEW.ST"


def test_same_isin_with_new_lei_keeps_same_entity_and_new_observation(
    tmp_path,
):
    identity = _identity(tmp_path)

    first = identity.discover(
        isin="SE0000000003",
        issuer="Example AB",
        lei="LEI000000000000000003",
        source="test",
        observed_date="2026-01-01",
        status="observed",
    )

    second = identity.discover(
        isin="SE0000000003",
        issuer="Example AB",
        lei="LEI000000000000000004",
        source="test",
        observed_date="2026-02-01",
        status="observed",
    )

    assert first["entity_id"] == second["entity_id"]
    assert len(_records(identity)) == 2

    assert _records(identity)[0]["lei"] == (
        "LEI000000000000000003"
    )
    assert _records(identity)[1]["lei"] == (
        "LEI000000000000000004"
    )


def test_new_lei_with_existing_isin_bridges_to_existing_entity(
    tmp_path,
):
    identity = _identity(tmp_path)

    first = identity.discover(
        isin="SE0000000004",
        issuer="Example AB",
        lei="LEI000000000000000005",
        source="test",
        observed_date="2026-01-01",
        status="observed",
    )

    second = identity.discover(
        isin="SE0000000004",
        issuer="Example AB",
        lei="LEI000000000000000006",
        source="test",
        observed_date="2026-02-01",
        status="observed",
    )

    assert second["entity_id"] == first["entity_id"]

    assert len(identity.entity_registry.records) == 1
    assert len(_records(identity)) == 2


def test_same_observation_is_idempotent(
    tmp_path,
):
    identity = _identity(tmp_path)

    first = identity.discover(
        isin="SE0000000005",
        issuer="Example AB",
        lei="LEI000000000000000007",
        yahoo_symbol="EXAMPLE.ST",
        source="test",
        observed_date="2026-01-01",
        status="observed",
    )

    second = identity.discover(
        isin="SE0000000005",
        issuer="Example AB",
        lei="LEI000000000000000007",
        yahoo_symbol="EXAMPLE.ST",
        source="test",
        observed_date="2026-01-01",
        status="observed",
    )

    assert first == second
    assert len(_records(identity)) == 1
    assert len(identity.entity_registry.records) == 1


def test_same_isin_cannot_be_assigned_to_different_entity(
    tmp_path,
):
    identity = _identity(tmp_path)

    identity.discover(
        isin="SE0000000006",
        issuer="Example AB",
        lei="LEI000000000000000008",
        source="test",
        observed_date="2026-01-01",
        status="observed",
    )

    identity.entity_registry.add(
        entity_id="ENT-000002",
        legal_name="Another AB",
        observed_date="2026-01-01",
        source="test",
    )
    identity.entity_registry.save()

    conflicts = identity.validate_observation(
        {
            "entity_id": "ENT-000002",
            "isin": "SE0000000006",
            "issuer": "Another AB",
            "lei": "LEI000000000000000009",
            "source": "test",
            "status": "observed",
            "observed_date": "2026-02-01",
        }
    )

    assert len(conflicts) == 1
    assert conflicts[0].identifier_type == "isin"
    assert conflicts[0].identifier == "SE0000000006"
    assert conflicts[0].existing_entity_ids == (
        "ENT-000001",
    )


def test_same_lei_cannot_be_assigned_to_different_entity(
    tmp_path,
):
    identity = _identity(tmp_path)

    identity.discover(
        isin="SE0000000007",
        issuer="Example AB",
        lei="LEI000000000000000010",
        source="test",
        observed_date="2026-01-01",
        status="observed",
    )

    identity.entity_registry.add(
        entity_id="ENT-000002",
        legal_name="Another AB",
        observed_date="2026-01-01",
        source="test",
    )
    identity.entity_registry.save()

    conflicts = identity.validate_observation(
        {
            "entity_id": "ENT-000002",
            "isin": "SE0000000008",
            "issuer": "Another AB",
            "lei": "LEI000000000000000010",
            "source": "test",
            "status": "observed",
            "observed_date": "2026-02-01",
        }
    )

    assert len(conflicts) == 1
    assert conflicts[0].identifier_type == "lei"
    assert conflicts[0].identifier == (
        "LEI000000000000000010"
    )
    assert conflicts[0].existing_entity_ids == (
        "ENT-000001",
    )
