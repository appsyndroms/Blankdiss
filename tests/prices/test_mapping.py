import json

from prices import mapping
from prices import mapping_resolution


def test_explicit_isin_mapping_for_skf(
    monkeypatch,
):
    monkeypatch.setattr(
        mapping_resolution,
        "KNOWN_YAHOO_SYMBOLS_BY_ISIN",
        {
            "SE0000108227": "SKF-B.ST",
        },
    )

    symbol, source = (
        mapping_resolution.explicit_mapping(
            {
                "isin": "SE0000108227",
                "issuer": "Aktiebolaget SKF",
                "ticker": None,
                "exchange": None,
            }
        )
    )

    assert symbol == "SKF-B.ST"
    assert source == "known_isin"


def test_explicit_isin_mapping_takes_priority_over_name(
    monkeypatch,
):
    monkeypatch.setattr(
        mapping_resolution,
        "KNOWN_YAHOO_SYMBOLS_BY_ISIN",
        {
            "SE0000108227": "SKF-B.ST",
        },
    )

    monkeypatch.setattr(
        mapping_resolution,
        "KNOWN_YAHOO_SYMBOLS",
        {
            "aktiebolaget skf": "WRONG.ST",
        },
    )

    symbol, source = (
        mapping_resolution.explicit_mapping(
            {
                "isin": "SE0000108227",
                "issuer": "Aktiebolaget SKF",
                "ticker": None,
                "exchange": None,
            }
        )
    )

    assert symbol == "SKF-B.ST"
    assert source == "known_isin"


def test_unknown_isin_remains_unresolved(
    monkeypatch,
):
    monkeypatch.setattr(
        mapping_resolution,
        "KNOWN_YAHOO_SYMBOLS_BY_ISIN",
        {},
    )

    monkeypatch.setattr(
        mapping_resolution,
        "KNOWN_YAHOO_SYMBOLS",
        {},
    )

    symbol, source = (
        mapping_resolution.explicit_mapping(
            {
                "isin": "SE0099999999",
                "issuer": "Ett Okänt Bolag",
                "ticker": None,
                "exchange": None,
            }
        )
    )

    assert symbol is None
    assert source is None


def test_persist_known_yahoo_mapping_appends_new_mapping(
    tmp_path,
    monkeypatch,
):
    path = (
        tmp_path
        / "known_yahoo_symbols.jsonl"
    )

    monkeypatch.setattr(
        mapping,
        "KNOWN_YAHOO_SYMBOLS_PATH",
        path,
    )

    monkeypatch.setattr(
        mapping,
        "KNOWN_YAHOO_SYMBOLS",
        {},
    )

    persisted = (
        mapping._persist_known_yahoo_mapping(
            "Axvik Group AB",
            "AXVIK.ST",
        )
    )

    assert persisted is True

    records = [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    assert records == [
        {
            "name": "axvik",
            "yahoo_symbol": "AXVIK.ST",
        }
    ]


def test_persist_known_yahoo_mapping_is_idempotent(
    tmp_path,
    monkeypatch,
):
    path = (
        tmp_path
        / "known_yahoo_symbols.jsonl"
    )

    monkeypatch.setattr(
        mapping,
        "KNOWN_YAHOO_SYMBOLS_PATH",
        path,
    )

    monkeypatch.setattr(
        mapping,
        "KNOWN_YAHOO_SYMBOLS",
        {},
    )

    first = (
        mapping._persist_known_yahoo_mapping(
            "Axvik Group AB",
            "AXVIK.ST",
        )
    )

    second = (
        mapping._persist_known_yahoo_mapping(
            "Axvik Group AB",
            "AXVIK.ST",
        )
    )

    assert first is True
    assert second is False

    lines = path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 1


def test_persist_known_yahoo_mapping_does_not_overwrite_conflict(
    tmp_path,
    monkeypatch,
):
    path = (
        tmp_path
        / "known_yahoo_symbols.jsonl"
    )

    path.write_text(
        json.dumps(
            {
                "name": "axvik",
                "yahoo_symbol": "AXVIK.ST",
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        mapping,
        "KNOWN_YAHOO_SYMBOLS_PATH",
        path,
    )

    monkeypatch.setattr(
        mapping,
        "KNOWN_YAHOO_SYMBOLS",
        {
            "axvik": "AXVIK.ST",
        },
    )

    persisted = (
        mapping._persist_known_yahoo_mapping(
            "Axvik Group AB",
            "WRONG.ST",
        )
    )

    assert persisted is False

    records = [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    assert records == [
        {
            "name": "axvik",
            "yahoo_symbol": "AXVIK.ST",
        }
    ]


def test_persist_known_yahoo_mapping_uses_normalized_name(
    tmp_path,
    monkeypatch,
):
    path = (
        tmp_path
        / "known_yahoo_symbols.jsonl"
    )

    monkeypatch.setattr(
        mapping,
        "KNOWN_YAHOO_SYMBOLS_PATH",
        path,
    )

    monkeypatch.setattr(
        mapping,
        "KNOWN_YAHOO_SYMBOLS",
        {},
    )

    persisted = (
        mapping._persist_known_yahoo_mapping(
            "Nordic Iron Ore AB",
            "NIO.ST",
        )
    )

    assert persisted is True

    records = [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    assert records == [
        {
            "name": "nordic iron ore",
            "yahoo_symbol": "NIO.ST",
        }
    ]
