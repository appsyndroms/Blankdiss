from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from shared.instrument_identity import (
    IdentityContractError,
    InstrumentIdentity,
    normalize_identifier,
    normalize_isin,
    normalize_lei,
    normalize_text,
)

from prices.mapping_resolution import (
    mapping_status,
    migrate_existing_mapping,
    resolve_instrument,
)
from prices.mapping_utils import (
    clean_value,
    MAPPING_PATH,
)
from prices.fi_identity import (
    get_latest_fi_instruments,
    read_all_fi_data,
)


# ---------------------------------------------------------------------------
# Existing mapping
# ---------------------------------------------------------------------------

def load_instrument_map() -> dict[
    str,
    dict[str, Any],
]:
    if not MAPPING_PATH.exists():
        return {}

    try:
        with MAPPING_PATH.open(
            "r",
            encoding="utf-8",
        ) as handle:
            data = json.load(
                handle
            )

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return {}

    if not isinstance(data, dict):
        return {}

    return data


def save_instrument_map(
    mapping: dict[
        str,
        dict[str, Any],
    ],
) -> Any:
    MAPPING_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with MAPPING_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            mapping,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

        handle.write("\n")

    return MAPPING_PATH


# ---------------------------------------------------------------------------
# Build mapping
# ---------------------------------------------------------------------------

def build_instrument_map(
    *,
    existing: dict[
        str,
        dict[str, Any],
    ],
    fi_records: list[
        dict[str, Any]
    ],
) -> tuple[
    dict[str, dict[str, Any]],
    int,
    int,
]:
    fi_instruments = (
        get_latest_fi_instruments(
            fi_records
        )
    )

    mapping = migrate_existing_mapping(
        existing,
        fi_instruments,
    )

    new_mappings = 0
    unresolved = 0

    needs_mapping = [
        instrument
        for instrument in fi_instruments
        if (
            instrument["map_key"] not in mapping
            or not clean_value(
                mapping[
                    instrument["map_key"]
                ].get("yahoo_symbol")
            )
        )
    ]

    print(
        "Mappning: "
        f"{len(needs_mapping)} instrument "
        "behöver Yahoo-mappning."
    )

    diagnostic_counts = {
        "historical_isin": 0,
        "historical_lei": 0,
        "no_candidates": 0,
        "all_wrong_exchange": 0,
        "no_equity_candidates": 0,
        "best_score_below_threshold": 0,
        "multiple_close_candidates": 0,
        "other": 0,
    }

    for index, instrument in enumerate(
        needs_mapping,
        start=1,
    ):
        key = instrument["map_key"]

        issuer = (
            instrument.get("issuer")
            or "?"
        )

        isin = instrument.get("isin")
        ticker = instrument.get("ticker")
        exchange = instrument.get("exchange")

        symbol, source, diagnostics = (
            resolve_instrument(
                instrument
            )
        )

        if symbol:
            mapping[key] = {
                "map_key": key,
                "isin": isin,
                "lei": instrument.get("lei"),
                "issuer": issuer,
                "ticker": ticker,
                "exchange": exchange,
                "yahoo_symbol": symbol,
                "mapping_source": source,
                "mapping_confidence": (
                    "high"
                    if source
                    in {
                        "known_isin",
                        "known_name",
                        "historical_isin",
                        "historical_lei",
                        "fi_ticker_exchange",
                    }
                    else "medium"
                ),
                "mapping_status": "mapped",
            }

            new_mappings += 1

            if source in diagnostic_counts:
                diagnostic_counts[source] += 1

            print(
                "Mappning: hittad - "
                f"{issuer} "
                f"ticker={ticker} "
                f"exchange={exchange} "
                f"isin={isin} "
                f"-> {symbol} "
                f"[{index}/"
                f"{len(needs_mapping)}]"
            )

            continue

        old = mapping.get(
            key,
            {},
        )

        mapping[key] = {
            **old,
            "map_key": key,
            "isin": isin,
            "lei": instrument.get("lei"),
            "issuer": issuer,
            "ticker": ticker,
            "exchange": exchange,
            "yahoo_symbol": None,
            "mapping_source": None,
            "mapping_confidence": None,
            "mapping_status": mapping_status(
                instrument
            ),
        }

        unresolved += 1

        print(
            "Mappning: ej hittad - "
            f"{issuer} "
            f"ticker={ticker} "
            f"exchange={exchange} "
            f"isin={isin} "
            f"[{index}/"
            f"{len(needs_mapping)}]"
        )

        if diagnostics:
            print_mapping_diagnostic(
                instrument,
                diagnostics,
            )

            reason = diagnostics.get(
                "reason"
            )

            if reason in diagnostic_counts:
                diagnostic_counts[
                    reason
                ] += 1
            else:
                diagnostic_counts[
                    "other"
                ] += 1

    identity = InstrumentIdentity()

    identity_sync_count = (
        sync_identity_from_yahoo_mapping(
            identity=identity,
            fi_instruments=fi_instruments,
            mapping=mapping,
        )
    )

    print(
        "Identity: "
        f"{identity_sync_count} "
        "nya/ändrade observationer."
    )

    print(
        "Mappning diagnostik:"
    )

    print(
        "  historisk ISIN-mappning: "
        f"{diagnostic_counts['historical_isin']}"
    )

    print(
        "  historisk LEI-mappning: "
        f"{diagnostic_counts['historical_lei']}"
    )

    print(
        "  inga Yahoo-kandidater: "
        f"{diagnostic_counts['no_candidates']}"
    )

    print(
        "  alla kandidater från fel börs: "
        f"{diagnostic_counts['all_wrong_exchange']}"
    )

    print(
        "  inga användbara aktiekandidater: "
        f"{diagnostic_counts['no_equity_candidates']}"
    )

    print(
        "  bästa score under tröskel: "
        f"{diagnostic_counts['best_score_below_threshold']}"
    )

    print(
        "  flera nära kandidater: "
        f"{diagnostic_counts['multiple_close_candidates']}"
    )

    print(
        "  övrigt: "
        f"{diagnostic_counts['other']}"
    )

    print(
        "Mappning: "
        f"{new_mappings} nya, "
        f"{unresolved} olösta, "
        f"{len(mapping)} totalt"
    )

    return (
        mapping,
        new_mappings,
        unresolved,
    )


def print_mapping_diagnostic(
    instrument: dict[str, Any],
    diagnostics: dict[str, Any],
) -> None:
    issuer = (
        instrument.get("issuer")
        or "?"
    )

    print(
        "Mappning diagnostik: "
        f"{issuer}"
    )

    print(
        "  identitet: "
        f"isin={instrument.get('isin')} "
        f"lei={instrument.get('lei')} "
        f"ticker={instrument.get('ticker')} "
        f"exchange={instrument.get('exchange')}"
    )

    for query in diagnostics.get(
        "queries",
        [],
    ):
        print(
            "  sökning: "
            f"{query['source']} "
            f"{query['query']!r} "
            f"-> "
            f"{query['result_count']} "
            "Yahoo-resultat"
        )

    rejected = diagnostics.get(
        "exchange_rejected",
        [],
    )

    if rejected:
        unique_rejected: dict[
            str,
            dict[str, Any],
        ] = {}

        for item in rejected:
            unique_rejected[
                item["symbol"]
            ] = item

        print(
            "  fel börs: "
            + ", ".join(
                (
                    f"{item['symbol']} "
                    f"({item['exchange']})"
                )
                for item in unique_rejected.values()
            )
        )

    accepted = diagnostics.get(
        "accepted_candidates",
        [],
    )

    if accepted:
        print(
            "  accepterade kandidater: "
            + ", ".join(
                (
                    f"{item['symbol']} "
                    f"score={item['score']:.3f}"
                )
                for item in accepted
            )
        )

    print(
        "  diagnos: "
        f"{diagnostics.get('reason')}"
    )


# ---------------------------------------------------------------------------
# Identity integration
# ---------------------------------------------------------------------------

def sync_identity_from_yahoo_mapping(
    *,
    identity: InstrumentIdentity,
    fi_instruments: list[dict[str, Any]],
    mapping: dict[str, dict[str, Any]],
) -> int:
    """
    Synkroniserar FI-identitet med Yahoo-mappningen.

    Yahoo-mappningen är den normala integrationspunkten för identity.
    Endast nya eller förändrade identitetsobservationer skrivs.

    Om en identity-konflikt uppstår återställs identity-filerna till
    läget före synkningen så att en körning inte lämnar halv data.
    """

    aliases_path = identity.aliases_path
    entities_path = identity.entity_registry.path

    aliases_backup = (
        aliases_path.read_bytes()
        if aliases_path.exists()
        else None
    )

    entities_backup = (
        entities_path.read_bytes()
        if entities_path.exists()
        else None
    )

    changed = 0

    try:
        for instrument in fi_instruments:
            key = instrument["map_key"]

            mapped = mapping.get(
                key
            )

            if not isinstance(
                mapped,
                dict,
            ):
                continue

            yahoo_symbol = clean_value(
                mapped.get(
                    "yahoo_symbol"
                )
            )

            if not yahoo_symbol:
                continue

            isin = clean_value(
                mapped.get("isin")
                or instrument.get("isin")
            )

            lei = clean_value(
                mapped.get("lei")
                or instrument.get("lei")
            )

            issuer = clean_value(
                mapped.get("issuer")
                or instrument.get("issuer")
            )

            ticker = clean_value(
                mapped.get("ticker")
                or instrument.get("ticker")
            )

            observed_date = (
                instrument.get("date")
                or date.today().isoformat()
            )

            existing = identity.resolve(
                isin=isin,
                lei=lei,
                issuer=issuer,
                ticker=ticker,
                yahoo_symbol=yahoo_symbol,
            )

            if (
                existing is not None
                and _same_identity_observation(
                    existing.record,
                    isin=isin,
                    lei=lei,
                    issuer=issuer,
                    ticker=ticker,
                    yahoo_symbol=yahoo_symbol,
                )
            ):
                continue

            identity.discover(
                isin=isin,
                issuer=issuer,
                lei=lei,
                ticker=ticker,
                yahoo_symbol=yahoo_symbol,
                source="FI_YAHOO",
                observed_date=observed_date,
                relation="observed",
                status="observed",
            )

            changed += 1

    except IdentityContractError:
        _restore_identity_file(
            aliases_path,
            aliases_backup,
        )

        _restore_identity_file(
            entities_path,
            entities_backup,
        )

        raise

    except Exception:
        _restore_identity_file(
            aliases_path,
            aliases_backup,
        )

        _restore_identity_file(
            entities_path,
            entities_backup,
        )

        raise

    return changed


def _same_identity_observation(
    record: dict[str, Any],
    *,
    isin: str | None,
    lei: str | None,
    issuer: str | None,
    ticker: str | None,
    yahoo_symbol: str | None,
) -> bool:
    return (
        normalize_isin(
            record.get("isin")
        )
        == normalize_isin(
            isin
        )
        and normalize_lei(
            record.get("lei")
        )
        == normalize_lei(
            lei
        )
        and normalize_text(
            record.get("issuer")
        )
        == normalize_text(
            issuer
        )
        and normalize_identifier(
            record.get("ticker")
        )
        == normalize_identifier(
            ticker
        )
        and normalize_identifier(
            record.get("yahoo_symbol")
        )
        == normalize_identifier(
            yahoo_symbol
        )
    )


def _restore_identity_file(
    path: Path,
    backup: bytes | None,
) -> None:
    if backup is None:
        if path.exists():
            path.unlink()

        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_bytes(
        backup
    )


# ---------------------------------------------------------------------------
# Yahoo symbols
# ---------------------------------------------------------------------------

def get_yahoo_symbols(
    mapping: dict[
        str,
        dict[str, Any],
    ],
) -> list[
    dict[str, Any]
]:
    instruments: list[
        dict[str, Any]
    ] = []

    for item in mapping.values():
        if not isinstance(
            item,
            dict,
        ):
            continue

        symbol = clean_value(
            item.get(
                "yahoo_symbol"
            )
        )

        if not symbol:
            continue

        instruments.append(
            item
        )

    unique: dict[
        str,
        dict[str, Any],
    ] = {}

    for item in instruments:
        symbol = item[
            "yahoo_symbol"
        ]

        unique[symbol] = item

    return [
        unique[symbol]
        for symbol in sorted(
            unique
        )
    ]


# ---------------------------------------------------------------------------
# Compatibility exports
# ---------------------------------------------------------------------------

__all__ = [
    "build_instrument_map",
    "get_latest_fi_instruments",
    "get_yahoo_symbols",
    "load_instrument_map",
    "read_all_fi_data",
    "save_instrument_map",
    "sync_identity_from_yahoo_mapping",
]
