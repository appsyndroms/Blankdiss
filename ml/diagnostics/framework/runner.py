from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable

from .base import DiagnosticExperiment
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
    Registrerar ett legacy-experiment.

    Legacy-modulen måste exponera run() eller main().
    """

    def factory(**kwargs):
        module = importlib.import_module(
            module_name
        )

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


def _find_experiment_class(
    module_name: str,
) -> type[DiagnosticExperiment]:
    module = importlib.import_module(
        module_name
    )

    experiment_classes = [
        value
        for value in vars(module).values()
        if (
            isinstance(value, type)
            and issubclass(
                value,
                DiagnosticExperiment,
            )
            and value is not DiagnosticExperiment
        )
    ]

    if not experiment_classes:
        raise RuntimeError(
            f"Modulen {module_name} saknar en "
            "DiagnosticExperiment-klass."
        )

    if len(experiment_classes) > 1:
        names = ", ".join(
            cls.__name__
            for cls in experiment_classes
        )

        raise RuntimeError(
            f"Modulen {module_name} innehåller flera "
            "DiagnosticExperiment-klasser: "
            f"{names}"
        )

    return experiment_classes[0]


def load_experiment(
    module_name: str,
    **kwargs: Any,
) -> DiagnosticExperiment:
    """
    Laddar den enda DiagnosticExperiment-klassen
    från registry-modulen.
    """

    experiment_class = _find_experiment_class(
        module_name
    )

    instance = experiment_class(
        **kwargs
    )

    if not isinstance(
        instance,
        DiagnosticExperiment,
    ):
        raise TypeError(
            f"{experiment_class.__name__} i "
            f"{module_name} är inte ett "
            "DiagnosticExperiment."
        )

    return instance


def list_experiments() -> list[ExperimentSpec]:
    return sorted(
        _REGISTRY.values(),
        key=lambda x: x.name,
    )


def get_experiment(
    name: str,
) -> ExperimentSpec:
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

    instance = spec.factory(
        **kwargs
    )

    if isinstance(
        instance,
        DiagnosticExperiment,
    ):
        result = instance.execute(
            context
        )

        if print_output:
            print_result(result)

        return result

    if context is not None:
        kwargs.setdefault(
            "context",
            context,
        )

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
