def prepare_data(config: DiscoveryConfig) -> DiscoveryData:
    _validate_targets(config)
    _validate_signals(config)

    frame = load_features()

    snapshot_dates = pd.to_datetime(
        frame["snapshot_date"],
        errors="coerce",
    )

    discovery_end = pd.Timestamp(
        config.discovery_end_date
    )

    frame = frame.loc[
        snapshot_dates <= discovery_end
    ].copy()

    if frame.empty:
        raise ValueError(
            "Discovery-perioden innehåller inga feature-rader."
        )

    targets: dict[str, np.ndarray] = {}
    returns: dict[str, np.ndarray] = {}

    for target_name in config.targets:
        target = _target_map()[target_name]

        target_values = build_target(
            frame,
            target,
        )

        targets[target_name] = np.asarray(
            target_values,
            dtype=float,
        )

        return_column = _target_return_column(
            target
        )

        if return_column not in frame.columns:
            raise ValueError(
                f"Return column '{return_column}' for target "
                f"'{target_name}' is missing from feature frame."
            )

        returns[target_name] = pd.to_numeric(
            frame[return_column],
            errors="coerce",
        ).to_numpy(dtype=float)

    signals: dict[str, np.ndarray] = {}

    for signal_name in config.signals:
        signals[signal_name] = np.asarray(
            build_signal(
                frame,
                signal_name,
            ),
            dtype=float,
        )

    stress_signals: dict[str, np.ndarray] = {}

    for stress_feature in config.stress_features:
        stress_signals[stress_feature] = np.asarray(
            build_signal(
                frame,
                stress_feature,
            ),
            dtype=float,
        )

    windows = _build_window_masks(
        frame
    )

    return DiscoveryData(
        frame=frame,
        targets=targets,
        signals=signals,
        stress_signals=stress_signals,
        windows=windows,
        returns=returns,
    )
