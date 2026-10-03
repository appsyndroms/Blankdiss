from pathlib import Path

import pytest

from felsokning import import_instrument_entities as importer
from shared.entity_identity import EntityRegistry
from shared.instrument_identity import IdentityContractError


def _discovery(
    *,
    lei: str = "LEI000000000000000001",
    isin: str = "SE0000000001",
    issuer: str = "Example AB",
    yahoo_symbol: str = "EXAMPLE.ST",
    status: str = "gleif_lei_without_entity",
    entity_ids: list[str] | None = None,
) -> dict:
    return {
        "source": {
            "mapping_date": "2026-10-01",
        },
        "instruments": [
            {
                "isin": isin,
                "issuer": issuer,
                "yahoo_symbol": yahoo_symbol,
                "gleif_leis": [lei],
                "entity_ids": entity_ids or [],
                "status": status,
                "instrument_lei": lei,
            }
        ],
    }


def _alias(
    *,
    entity_id: str,
    isin: str,
    lei: str,
    issuer: str = "Example AB",
    yahoo_symbol: str = "EXAMPLE.ST",
    observed_date: str = "2026-09-01",
    source: str = "test",
) -> dict:
    return {
        "entity_id": entity_id,
        "isin": isin,
        "issuer": issuer,
        "yahoo_symbol": yahoo_symbol,
        "lei": lei,
        "valid_from": None,
        "valid_to": None,
        "relation": "observed",
        "source": source,
        "status": "observed",
        "observed_date": observed_date,
    }


def _registry(
    tmp_path: Path,
) -> EntityRegistry:
    return EntityRegistry(
        path=tmp_path / "instrument_entities.jsonl"
    )


def _patch_identity_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        importer,
        "ALIASES_PATH",
        tmp_path / "instrument_aliases.jsonl",
    )
    monkeypatch.setattr(
        importer,
        "ENTITY_REGISTRY_PATH",
        tmp_path / "instrument_entities.jsonl",
    )


def test_new_entity_candidate_passes_validation(
    tmp_path,
    monkeypatch,
):
    _patch_identity_paths(
        monkeypatch,
        tmp_path,
    )

    aliases = []
    registry = _registry(
        tmp_path
    )

    entities, import_records, summary = (
        importer._build_import_plan(
            _discovery(),
            aliases,
            registry,
        )
    )

    assert len(entities) == 1
    assert entities[0]["status"] == (
        "new_entity_candidate"
    )
    assert entities[0]["entity_id"] == (
        "ENT-000001"
    )

    assert len(import_records) == 1
    assert import_records[0]["entity_id"] == (
        "ENT-000001"
    )

    assert summary["new_entity_candidates"] == 1
    assert summary["conflicting_entities"] == 0

    importer._validate_import_plan(
        aliases,
        entities,
        import_records,
    )


def test_entity_resolved_by_gleif_reuses_existing_entity(
    tmp_path,
    monkeypatch,
):
    _patch_identity_paths(
        monkeypatch,
        tmp_path,
    )

    lei = "LEI000000000000000010"
    isin = "SE0000000010"

    aliases = [
        _alias(
            entity_id="ENT-000001",
            isin=isin,
            lei=lei,
            source="GLEIF_ANNA",
            observed_date="2026-10-01",
        )
    ]

    registry = _registry(
        tmp_path
    )

    registry.add(
        entity_id="ENT-000001",
        legal_name="Example AB",
        observed_date="2026-10-01",
        source="test",
    )

    registry.save()

    entities, import_records, summary = (
        importer._build_import_plan(
            _discovery(
                lei=lei,
                isin=isin,
                status="entity_resolved_by_gleif",
                entity_ids=["ENT-000001"],
            ),
            aliases,
            registry,
        )
    )

    assert len(entities) == 1
    assert entities[0]["status"] == (
        "existing_entity"
    )
    assert entities[0]["entity_id"] == (
        "ENT-000001"
    )

    assert import_records == []
    assert summary["existing_entities_reused"] == 1
    assert summary["already_present"] == 1

    importer._validate_import_plan(
        aliases,
        entities,
        import_records,
    )


def test_multiple_instruments_with_same_lei_share_one_entity(
    tmp_path,
    monkeypatch,
):
    _patch_identity_paths(
        monkeypatch,
        tmp_path,
    )

    lei = "LEI000000000000000011"

    existing_isin = "SE0000000011"
    new_isin = "SE0000000012"

    aliases = [
        _alias(
            entity_id="ENT-000001",
            isin=existing_isin,
            lei=lei,
            issuer="Example AB",
            yahoo_symbol="EXAMPLE-A.ST",
            source="GLEIF_ANNA",
            observed_date="2026-10-01",
        )
    ]

    registry = _registry(
        tmp_path
    )

    registry.add(
        entity_id="ENT-000001",
        legal_name="Example AB",
        observed_date="2026-10-01",
        source="test",
    )

    registry.save()

    discovery = {
        "source": {
            "mapping_date": "2026-10-01",
        },
        "instruments": [
            {
                "isin": existing_isin,
                "issuer": "Example AB",
                "yahoo_symbol": "EXAMPLE-A.ST",
                "gleif_leis": [lei],
                "entity_ids": ["ENT-000001"],
                "status": "entity_resolved_by_gleif",
                "instrument_lei": lei,
            },
            {
                "isin": new_isin,
                "issuer": "Example AB",
                "yahoo_symbol": "EXAMPLE-B.ST",
                "gleif_leis": [lei],
                "entity_ids": ["ENT-000001"],
                "status": "entity_resolved_by_gleif",
                "instrument_lei": lei,
            },
        ],
    }

    entities, import_records, summary = (
        importer._build_import_plan(
            discovery,
            aliases,
            registry,
        )
    )

    # Två instrument med samma LEI ska ge EN entity-grupp.
    assert len(entities) == 1

    assert entities[0]["status"] == (
        "existing_entity"
    )
    assert entities[0]["entity_id"] == (
        "ENT-000001"
    )
    assert entities[0]["lei"] == lei
    assert entities[0]["instrument_count"] == 2

    # Första instrumentet finns redan.
    # Det andra blir en ny observation på samma entity.
    assert len(import_records) == 1
    assert import_records[0]["entity_id"] == (
        "ENT-000001"
    )
    assert import_records[0]["isin"] == new_isin
    assert import_records[0]["lei"] == lei

    assert summary["candidate_lei_groups"] == 1
    assert summary["existing_entities_reused"] == 1
    assert summary["new_entity_candidates"] == 0
    assert summary["already_present"] == 1
    assert summary["new_alias_records"] == 1
    assert summary["conflicting_entities"] == 0

    importer._validate_import_plan(
        aliases,
        entities,
        import_records,
    )


def test_existing_isin_bridges_to_existing_entity(
    tmp_path,
    monkeypatch,
):
    _patch_identity_paths(
        monkeypatch,
        tmp_path,
    )

    aliases = [
        _alias(
            entity_id="ENT-000001",
            isin="SE0000000002",
            lei="LEI000000000000000002",
        )
    ]

    registry = _registry(
        tmp_path
    )

    registry.add(
        entity_id="ENT-000001",
        legal_name="Example AB",
        observed_date="2026-09-01",
        source="test",
    )

    registry.save()

    entities, import_records, summary = (
        importer._build_import_plan(
            _discovery(
                lei="LEI000000000000000003",
                isin="SE0000000002",
            ),
            aliases,
            registry,
        )
    )

    assert len(entities) == 1
    assert entities[0]["status"] == (
        "existing_entity_via_isin"
    )
    assert entities[0]["entity_id"] == (
        "ENT-000001"
    )

    assert len(import_records) == 1
    assert import_records[0]["entity_id"] == (
        "ENT-000001"
    )
    assert import_records[0]["lei"] == (
        "LEI000000000000000003"
    )

    assert summary["conflicting_entities"] == 0

    importer._validate_import_plan(
        aliases,
        entities,
        import_records,
    )


def test_conflicting_isin_stops_import_validation(
    tmp_path,
    monkeypatch,
):
    _patch_identity_paths(
        monkeypatch,
        tmp_path,
    )

    aliases = [
        _alias(
            entity_id="ENT-000001",
            isin="SE0000000003",
            lei="LEI000000000000000004",
        ),
        _alias(
            entity_id="ENT-000002",
            isin="SE0000000003",
            lei="LEI000000000000000005",
        ),
    ]

    registry = _registry(
        tmp_path
    )

    registry.add(
        entity_id="ENT-000001",
        legal_name="Example AB",
        observed_date="2026-09-01",
        source="test",
    )

    registry.add(
        entity_id="ENT-000002",
        legal_name="Another AB",
        observed_date="2026-09-01",
        source="test",
    )

    registry.save()

    entities, import_records, summary = (
        importer._build_import_plan(
            _discovery(
                lei="LEI000000000000000006",
                isin="SE0000000003",
            ),
            aliases,
            registry,
        )
    )

    assert len(entities) == 1
    assert entities[0]["status"] == "conflict"
    assert entities[0]["entity_ids"] == [
        "ENT-000001",
        "ENT-000002",
    ]

    assert import_records == []
    assert summary["conflicting_entities"] == 1

    with pytest.raises(
        IdentityContractError,
        match="identity-konflikter",
    ):
        importer._validate_import_plan(
            aliases,
            entities,
            import_records,
        )


def test_conflicting_lei_stops_import_validation(
    tmp_path,
    monkeypatch,
):
    _patch_identity_paths(
        monkeypatch,
        tmp_path,
    )

    aliases = [
        _alias(
            entity_id="ENT-000001",
            isin="SE0000000004",
            lei="LEI000000000000000007",
        ),
        _alias(
            entity_id="ENT-000002",
            isin="SE0000000005",
            lei="LEI000000000000000007",
        ),
    ]

    registry = _registry(
        tmp_path
    )

    registry.add(
        entity_id="ENT-000001",
        legal_name="Example AB",
        observed_date="2026-09-01",
        source="test",
    )

    registry.add(
        entity_id="ENT-000002",
        legal_name="Another AB",
        observed_date="2026-09-01",
        source="test",
    )

    registry.save()

    entities, import_records, summary = (
        importer._build_import_plan(
            _discovery(
                lei="LEI000000000000000007",
                isin="SE0000000006",
            ),
            aliases,
            registry,
        )
    )

    assert len(entities) == 1
    assert entities[0]["status"] == "conflict"
    assert entities[0]["entity_ids"] == [
        "ENT-000001",
        "ENT-000002",
    ]

    assert import_records == []
    assert summary["conflicting_entities"] == 1

    with pytest.raises(
        IdentityContractError,
        match="identity-konflikter",
    ):
        importer._validate_import_plan(
            aliases,
            entities,
            import_records,
        )


def test_exact_duplicate_is_skipped(
    tmp_path,
    monkeypatch,
):
    _patch_identity_paths(
        monkeypatch,
        tmp_path,
    )

    aliases = [
        _alias(
            entity_id="ENT-000001",
            isin="SE0000000007",
            lei="LEI000000000000000008",
            issuer="Example AB",
            yahoo_symbol="EXAMPLE.ST",
            observed_date="2026-10-01",
            source="GLEIF_ANNA",
        )
    ]

    registry = _registry(
        tmp_path
    )

    registry.add(
        entity_id="ENT-000001",
        legal_name="Example AB",
        observed_date="2026-10-01",
        source="GLEIF_ANNA",
    )

    registry.save()

    entities, import_records, summary = (
        importer._build_import_plan(
            _discovery(
                lei="LEI000000000000000008",
                isin="SE0000000007",
                issuer="Example AB",
                yahoo_symbol="EXAMPLE.ST",
            ),
            aliases,
            registry,
        )
    )

    assert len(entities) == 1
    assert entities[0]["status"] == (
        "existing_entity"
    )
    assert entities[0]["entity_id"] == (
        "ENT-000001"
    )

    assert import_records == []
    assert summary["conflicting_entities"] == 0
    assert summary["already_present"] == 1

    importer._validate_import_plan(
        aliases,
        entities,
        import_records,
    )
