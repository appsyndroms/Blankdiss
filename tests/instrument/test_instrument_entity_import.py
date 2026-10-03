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
) -> dict:
    return {
        "source": {
            "mapping_date": "2026-10-01",
        },
        "candidate_entities": [
            {
                "lei": lei,
                "instruments": [
                    {
                        "isin": isin,
                        "issuer": issuer,
                        "yahoo_symbol": yahoo_symbol,
                    }
                ],
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
