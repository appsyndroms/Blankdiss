from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

SPEC_DIR = (
    ROOT
    / "ml"
    / "research"
    / "specs"
)

RUNS_DIR = (
    ROOT
    / "data"
    / "processed"
    / "ml"
    / "research"
    / "spec_runs"
)

DISCOVERY_DIR = (
    ROOT
    / "ml"
    / "research"
    / "discovery"
)

STATE_DIR = (
    ROOT
    / "data"
    / "ai_lab"
    / "adaptive_research"
)

STATE_PATH = (
    STATE_DIR
    / "state.json"
)


SOURCE_SPEC_ID = (
    "momentum_si_regime_walk_forward"
)

LOCKED_CONFIRMATION_ID = (
    "momentum_si_prospective_confirmation"
)

ADAPTIVE_PREFIX = (
    "adaptive_momentum_si_"
)


# Fixed, predeclared adaptive parameter space.
#
# This order is established before observing adaptive results.
ADAPTIVE_FRACTIONS = (
    0.30,
    0.20,
    0.10,
    0.05,
    0.025,
)


MIN_VALID_N = 100

SIGN_EPSILON = 1e-12


# Declared extension families.
#
# The engine does not invent arbitrary Python at runtime.
EXTENSION_FAMILIES = (
    "target_profile",
)
