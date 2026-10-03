from pathlib import Path

import pytest

from prices.mapping import (
    sync_identity_from_yahoo_mapping,
)
from shared.instrument_identity import (
    IdentityContractError,
    InstrumentIdentity,
)


def _identity(tmp_path: Path) -> InstrumentIdentity:
    return InstrumentIdentity(
        instrument_map_path=(
            tmp_path
            / "instrument_map.json"
        ),
        aliases_path=(
            tmp_path
            / "instrument_aliases.jsonl"
        ),
        entity_registry_path=(
            tmp_path
            / "instrument_entities.jsonl"
        ),
    )


def _instrument(
    *,
    isin: str,
    lei: str,
    issuer: str,
    ticker: str,
    date: str = "2026-10-03",
) -> dict:
    return {
        "map_key": f"ISIN:{isin}",
        "isin": isin,
        "lei": lei,
        "issuer": issuer,
        "ticker": ticker,
        "exchange": "STO",
        "date": date,
    }


def _mapping(
    *,
    isin: str,
    lei: str,
    issuer: str,
    ticker: str,
    yahoo_symbol: str,
) -> dict:
    key = f"ISIN:{isin}"

    return {
        key: {
            "map_key": key,
            "isin": isin,
            "lei": lei,
            "issuer": issuer,
            "ticker": ticker,
            "exchange": "STO",
            "yahoo_symbol": yahoo_symbol,
            "mapping_status": "mapped",
        }
    }


def test_yahoo_mapping_creates_identity_observation(
    tmp_path,
):
    identity = _identity(tmp_path)

    isin = "SE0000001001"
    lei = "LEI000000000000001001"

    changed = sync_identity_from_yahoo_mapping(
        identity=identity,
        fi_instruments=[
            _instrument(
                isin=isin,
                lei=lei,
                issuer="Example AB",
                ticker="EXAMPLE",
            )
        ],
        mapping=_mapping(
            isin=isin,
            lei=lei,
            issuer="Example AB",
            ticker="EXAMPLE",
            yahoo_symbol="EXAMPLE.ST",
        ),
    )

    assert changed == 1
    assert len(identity.records) == 1
    assert len(
        identity.entity_registry.records
    ) == 1

    record = identity.records[0]

    assert record["isin"] == isin
    assert record["lei"] == lei
    assert record["issuer"] == "EXAMPLE AB"
    assert record["ticker"] == "EXAMPLE"
    assert record["yahoo_symbol"] == "EXAMPLE.ST"
    assert record["source"] == "FI_YAHOO"


def test_yahoo_mapping_is_idempotent(
    tmp_path,
):
    identity = _identity(tmp_path)

    isin = "SE0000001002"
    lei = "LEI000000000000001002"

    instruments = [
        _instrument(
            isin=isin,
            lei=lei,
            issuer="Example AB",
            ticker="EXAMPLE",
        )
    ]

    mapping = _mapping(
        isin=isin,
        lei=lei,
        issuer="Example AB",
        ticker="EXAMPLE",
        yahoo_symbol="EXAMPLE.ST",
    )

    first = sync_identity_from_yahoo_mapping(
        identity=identity,
        fi_instruments=instruments,
        mapping=mapping,
    )

    second = sync_identity_from_yahoo_mapping(
        identity=identity,
        fi_instruments=instruments,
        mapping=mapping,
    )

    assert first == 1
    assert second == 0
    assert len(identity.records) == 1
    assert len(
        identity.entity_registry.records
    ) == 1


def test_yahoo_symbol_change_keeps_entity_and_creates_observation(
    tmp_path,
):
    identity = _identity(tmp_path)

    isin = "SE0000001003"
    lei = "LEI000000000000001003"

    instrument = _instrument(
        isin=isin,
        lei=lei,
        issuer="Example AB",
        ticker="EXAMPLE",
    )

    first_mapping = _mapping(
        isin=isin,
        lei=lei,
        issuer="Example AB",
        ticker="EXAMPLE",
        yahoo_symbol="OLD.ST",
    )

    second_mapping = _mapping(
        isin=isin,
        lei=lei,
        issuer="Example AB",
        ticker="EXAMPLE",
        yahoo_symbol="NEW.ST",
    )

    sync_identity_from_yahoo_mapping(
        identity=identity,
        fi_instruments=[instrument],
        mapping=first_mapping,
    )

    first_entity_id = identity.records[0][
        "entity_id"
    ]

    changed = sync_identity_from_yahoo_mapping(
        identity=identity,
        fi_instruments=[instrument],
        mapping=second_mapping,
    )

    assert changed == 1
    assert len(identity.records) == 2

    assert identity.records[0][
        "entity_id"
    ] == first_entity_id

    assert identity.records[1][
        "entity_id"
    ] == first_entity_id

    assert identity.records[0][
        "yahoo_symbol"
    ] == "OLD.ST"

    assert identity.records[1][
        "yahoo_symbol"
    ] == "NEW.ST"


def test_new_lei_with_existing_isin_bridges_to_existing_entity(
    tmp_path,
):
    identity = _identity(tmp_path)

    isin = "SE0000001004"
    old_lei = "LEI000000000000001004"
    new_lei = "LEI000000000000001005"

    instrument = _instrument(
        isin=isin,
        lei=old_lei,
        issuer="Example AB",
        ticker="EXAMPLE",
    )

    sync_identity_from_yahoo_mapping(
        identity=identity,
        fi_instruments=[instrument],
        mapping=_mapping(
            isin=isin,
            lei=old_lei,
            issuer="Example AB",
            ticker="EXAMPLE",
            yahoo_symbol="EXAMPLE.ST",
        ),
    )

    first_entity_id = identity.records[0][
        "entity_id"
    ]

    instrument_with_new_lei = _instrument(
        isin=isin,
        lei=new_lei,
        issuer="Example AB",
        ticker="EXAMPLE",
    )

    changed = sync_identity_from_yahoo_mapping(
        identity=identity,
        fi_instruments=[
            instrument_with_new_lei
        ],
        mapping=_mapping(
            isin=isin,
            lei=new_lei,
            issuer="Example AB",
            ticker="EXAMPLE",
            yahoo_symbol="EXAMPLE.ST",
        ),
    )

    assert changed == 1
    assert len(identity.records) == 2
    assert len(
        identity.entity_registry.records
    ) == 1

    assert identity.records[1][
        "entity_id"
    ] == first_entity_id

    assert identity.records[1][
        "lei"
    ] == new_lei


def test_identity_conflict_rolls_back_all_changes(
    tmp_path,
):
    identity = _identity(tmp_path)

    first_isin = "SE0000001005"
    first_lei = "LEI000000000000001006"

    second_isin = "SE0000001006"
    second_lei = "LEI000000000000001007"

    sync_identity_from_yahoo_mapping(
        identity=identity,
        fi_instruments=[
            _instrument(
                isin=first_isin,
                lei=first_lei,
                issuer="First AB",
                ticker="FIRST",
            )
        ],
        mapping=_mapping(
            isin=first_isin,
            lei=first_lei,
            issuer="First AB",
            ticker="FIRST",
            yahoo_symbol="FIRST.ST",
        ),
    )

    sync_identity_from_yahoo_mapping(
        identity=identity,
        fi_instruments=[
            _instrument(
                isin=second_isin,
                lei=second_lei,
                issuer="Second AB",
                ticker="SECOND",
            )
        ],
        mapping=_mapping(
            isin=second_isin,
            lei=second_lei,
            issuer="Second AB",
            ticker="SECOND",
            yahoo_symbol="SECOND.ST",
        ),
    )

    aliases_before = (
        identity.aliases_path.read_bytes()
    )

    entities_before = (
        identity.entity_registry.path.read_bytes()
    )

    conflicting_instrument = _instrument(
        isin=first_isin,
        lei=second_lei,
        issuer="Conflicting AB",
        ticker="CONFLICT",
    )

    conflicting_mapping = _mapping(
        isin=first_isin,
        lei=second_lei,
        issuer="Conflicting AB",
        ticker="CONFLICT",
        yahoo_symbol="CONFLICT.ST",
    )

    with pytest.raises(
        IdentityContractError
    ):
        sync_identity_from_yahoo_mapping(
            identity=identity,
            fi_instruments=[
                conflicting_instrument
            ],
            mapping=conflicting_mapping,
        )

    assert (
        identity.aliases_path.read_bytes()
        == aliases_before
    )

    assert (
        identity.entity_registry.path.read_bytes()
        == entities_before
    )
