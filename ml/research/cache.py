def _build_window_masks(
    frame: pd.DataFrame,
) -> dict[str, dict[str, np.ndarray]]:
    dates = pd.to_datetime(
        frame["snapshot_date"],
        errors="coerce",
    )

    result: dict[
        str,
        dict[str, np.ndarray],
    ] = {}

    for index, window in enumerate(
        WALK_FORWARD_WINDOWS,
        start=1,
    ):
        train_end = pd.Timestamp(
            window.train_end
        )
        validation_end = pd.Timestamp(
            window.validation_end
        )
        test_end = pd.Timestamp(
            window.test_end
        )

        result[f"window_{index}"] = {
            "train": (
                dates <= train_end
            ).to_numpy(),

            "validation": (
                (dates > train_end)
                & (dates <= validation_end)
            ).to_numpy(),

            "test": (
                (dates > validation_end)
                & (dates <= test_end)
            ).to_numpy(),
        }

    return result
