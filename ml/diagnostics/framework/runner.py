from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable

from .base import DiagnosticExperiment, ExperimentResult
from .reporting import print_result


@dataclass
class ExperimentSpec:
    name: str
    factory: Callable[..., Any]
    description: str = ""
    legacy: bool = False


_REGISTRY: dict[str, ExperimentSpec] = {}


def register_experiment(
    name: str,
    factory: Callable[..., Any],
    description: str = "",
) -> None:
    if name in _REGISTRY:
        raise ValueError(
            f"Experiment '{name}' är redan registrerat."
        )

    _REGISTRY[name] = ExperimentSpec(
        name=name,
        factory=factory,
        description=description,
    )


def register_legacy(
    name: str,
    module_name: str,
    description: str = "",
) -> None:
    """
    Registrerar ett befintligt experiment utan att det
    behöver skrivas om direkt.

    Modulen måste ha en main() eller run()-funktion.
    """

    def factory(**kwargs):
        module = importlib.import_module(module_name)

        if hasattr(module, "run"):
            return module.run(**kwargs)

        if hasattr(module, "main"):
            return module.main(**kwargs)

        raise AttributeError(
            f"Legacy-experiment '{module_name}' saknar "
            "run() eller main()."
        )

    _REGISTRY[name] = ExperimentSpec(
        name=name,
        factory=factory,
        description=description,
        legacy=True,
    )


def list_experiments() -> list[ExperimentSpec]:
    return sorted(
        _REGISTRY.values(),
        key=lambda x: x.name,
    )


def get_experiment(name: str) -> ExperimentSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        available = ", ".join(
            sorted(_REGISTRY)
        )

        raise KeyError(
            f"Okänt experiment '{name}'. "
            f"Tillgängliga: {available}"
        )


def run_experiment(
    name: str,
    context=None,
    print_output: bool = True,
    **kwargs,
):
    spec = get_experiment(name)

    instance = spec.factory(**kwargs)

    if isinstance(instance, DiagnosticExperiment):
        result = instance.execute(context)

        if print_output:
            print_result(result)

        return result

    # Legacy support.
    if context is not None:
        kwargs.setdefault("context", context)

    result = instance

    if callable(result):
        result = result()

    return result


def run_many(
    names: list[str],
    context=None,
    print_output: bool = True,
    **kwargs,
):
    results = {}

    for name in names:
        results[name] = run_experiment(
            name,
            context=context,
            print_output=print_output,
            **kwargs,
        )

    return results
