"""
Presentation av ekonomiska ML-resultat.
"""

from __future__ import annotations

import html

from .formatting import format_pct


def _primary_strategy(
    experiment: dict,
) -> dict | None:
    for strategy in experiment.get(
        "strategies",
        [],
    ):
        fraction = strategy.get(
            "fraction"
        )

        if fraction is not None:
            if abs(
                float(fraction)
                - 0.01
            ) < 1e-12:
                return strategy

    return None


def build_ml_rows(
    economic_results: dict,
) -> str:
    experiments = economic_results.get(
        "experiments",
        [],
    )

    if not experiments:
        return """
        <tr>
            <td colspan="9" class="empty">
                Inga ML-resultat ännu.
            </td>
        </tr>
        """

    rows: list[str] = []

    for experiment in experiments:
        strategy = _primary_strategy(
            experiment
        )

        if strategy is None:
            continue

        feature_set = html.escape(
            str(
                experiment.get(
                    "feature_set",
                    "—",
                )
            )
        )

        target = html.escape(
            str(
                experiment.get(
                    "target",
                    "—",
                )
            )
        )

        task = html.escape(
            str(
                experiment.get(
                    "task",
                    "—",
                )
            )
        )

        model = html.escape(
            str(
                experiment.get(
                    "model",
                    "—",
                )
            )
        )

        direction = html.escape(
            str(
                experiment.get(
                    "direction",
                    "—",
                )
            )
        )

        rows.append(
            f"""
            <tr>
                <td>
                    <strong>{feature_set}</strong>
                </td>
                <td>
                    {target}
                </td>
                <td>
                    {task}
                </td>
                <td>
                    {model}
                </td>
                <td>
                    {direction}
                </td>
                <td>
                    {format_pct(
                        strategy.get(
                            "net_compounded_return"
                        )
                    )}
                </td>
                <td>
                    {format_pct(
                        strategy.get(
                            "benchmark_compounded_return"
                        )
                    )}
                </td>
                <td>
                    {format_pct(
                        strategy.get(
                            "excess_return_vs_benchmark"
                        )
                    )}
                </td>
                <td>
                    {format_pct(
                        strategy.get(
                            "max_drawdown"
                        )
                    )}
                </td>
            </tr>
            """
        )

    if not rows:
        return """
        <tr>
            <td colspan="9" class="empty">
                Inga kompletta ML-strategier ännu.
            </td>
        </tr>
        """

    return "\n".join(rows)
