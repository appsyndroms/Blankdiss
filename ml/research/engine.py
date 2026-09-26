    elif spec.analysis.type == "conditional_regime_comparison":
        if len(spec.signals) < 3:
            raise ValueError(
                "conditional_regime_comparison requires "
                "at least three signals."
            )

        if any(
            len(signal.bins) != 1
            for signal in spec.signals[:-1]
        ):
            raise ValueError(
                "conditional_regime_comparison requires exactly "
                "one bin for each baseline signal; only the last "
                "signal may have multiple bins."
            )

        baseline_fractions = tuple(
            signal.bins[0]
            for signal in spec.signals[:-1]
        )

        bootstrap = spec.analysis.bootstrap

        for incremental_fraction in spec.signals[-1].bins:
            fractions = (
                *baseline_fractions,
                incremental_fraction,
            )

            for target_name in spec.targets:
                for window_name in spec.windows:
                    for split_name in spec.splits:
                        results.append(
                            analyse_conditional_regime_comparison(
                                cache,
                                spec.signals,
                                target_name,
                                fractions,
                                window_name,
                                split_name,
                                bootstrap=bootstrap,
                                bootstrap_iterations=(
                                    spec.analysis
                                    .bootstrap_iterations
                                ),
                                spec_id=spec.id,
                            )
                        )
