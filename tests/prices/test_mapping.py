from prices.mapping import _explicit_mapping


def test_explicit_isin_mapping_for_skf():
    symbol, source = _explicit_mapping(
        {
            "isin": "SE0000108227",
            "issuer": "Aktiebolaget SKF",
            "ticker": None,
            "exchange": None,
        }
    )

    assert symbol == "SKF-B.ST"
    assert source == "known_isin"


def test_explicit_isin_mapping_takes_priority_over_name():
    symbol, source = _explicit_mapping(
        {
            "isin": "SE0000108227",
            "issuer": "Aktiebolaget SKF",
            "ticker": None,
            "exchange": None,
        }
    )

    assert symbol == "SKF-B.ST"
    assert source == "known_isin"


def test_unknown_isin_remains_unresolved():
    symbol, source = _explicit_mapping(
        {
            "isin": "SE0099999999",
            "issuer": "Ett Okänt Bolag",
            "ticker": None,
            "exchange": None,
        }
    )

    assert symbol is None
    assert source is None
