def attach_analysis_targets_to_oos(oos, data):
    required_oos = {
        "snapshot_date",
        "security_key",
        "prediction",
        "score",
    }
    missing_oos = required_oos - set(oos.columns)
    if missing_oos:
        raise ValueError(
            "OOS predictions are missing columns: "
            f"{sorted(missing_oos)}"
        )
    diagnostic_columns = [
        "snapshot_date",
        "security_key",
        "forward_return_5d",
        "target_down_5pct_5d",
        "target_up_5pct_5d",
        "target_abs_5pct_5d",
        "target_abs_10pct_5d",
        "price_volatility_20d",
        "volatility_60d",
        "volatility_20d_minus_60d",
        "volatility_20d_div_60d",
        "price_return_5d",
        "price_return_20d",
        "price_return_60d",
    ]
    missing_data = [
        column
        for column in diagnostic_columns
        if column not in data.columns
    ]
    if missing_data:
        raise ValueError(
            "Diagnostic data is missing columns: "
            f"{missing_data}"
        )
    oos = oos.copy()
    lookup = data[diagnostic_columns].copy()
    # ml.walk_forward serialiserar snapshot_date till
    # YYYY-MM-DD i OOS-resultatet. Normalisera även data-sidan
    # till samma representation före merge.
    oos["snapshot_date"] = pd.to_datetime(
        oos["snapshot_date"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    lookup["snapshot_date"] = pd.to_datetime(
        lookup["snapshot_date"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    lookup = lookup.drop_duplicates(
        subset=[
            "snapshot_date",
            "security_key",
        ],
        keep="last",
    )
    return oos.merge(
        lookup,
        on=[
            "snapshot_date",
            "security_key",
        ],
        how="left",
        validate="many_to_one",
        suffixes=("", "_data"),
    )
