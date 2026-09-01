"""Ablation modes for the Koi paper evaluation.

Two independent, boot-time switches (configured once by the runner, before the
first tick; never toggled mid-run):

    mechanism mode  "full" | "inert"
        "inert" removes the causal DAG/mechanism machinery from every decision:
        EIG is exactly 0, mechanism selection is replaced by one pass-through
        sentinel id (the planner's hard gates require a resolvable
        mechanism_id), confidence is never read, and the DAG tools and prompt
        sections are withheld from the LLM. Candidate enumeration, prediction,
        Tchebycheff/DRO/switch-cost scoring, and C0-C6 validation are untouched.

    learning mode  "online" | "frozen"
        "frozen" stops every update from observations: S3 (DRO residuals, Beta
        confidence, slow-loop knobs, CUSUM recalibration) is skipped, surrogate
        calibration/fusion loses its evidence input, dead-shape memory is
        disabled, and mechanism admission is refused. Evidence rows are still
        appended so runs stay analyzable; EIG still runs, from the frozen seed
        priors.
"""

MECHANISM_MODES = ("full", "inert")
LEARNING_MODES = ("online", "frozen")

_mechanism_mode = "full"
_learning_mode = "online"
_passthrough_mechanism_id: str | None = None


def configure_mechanism_mode(mode: str) -> None:
    """Set the mechanism-ablation mode. Call once at boot."""
    if not isinstance(mode, str):
        raise TypeError("mechanism mode must be a string")
    if mode not in MECHANISM_MODES:
        raise ValueError("mechanism mode must be one of: " + ", ".join(MECHANISM_MODES))
    global _mechanism_mode
    _mechanism_mode = mode


def configure_learning_mode(mode: str) -> None:
    """Set the learning-ablation mode. Call once at boot."""
    if not isinstance(mode, str):
        raise TypeError("learning mode must be a string")
    if mode not in LEARNING_MODES:
        raise ValueError("learning mode must be one of: " + ", ".join(LEARNING_MODES))
    global _learning_mode
    _learning_mode = mode


def set_passthrough_mechanism_id(mechanism_id: str) -> None:
    """Record the sentinel mechanism id registered for an inert-mode run."""
    if not isinstance(mechanism_id, str) or not mechanism_id:
        raise ValueError("pass-through mechanism id must be a non-empty string")
    global _passthrough_mechanism_id
    _passthrough_mechanism_id = mechanism_id


def mechanism_mode() -> str:
    """Return the current mechanism-ablation mode."""
    return _mechanism_mode


def learning_mode() -> str:
    """Return the current learning-ablation mode."""
    return _learning_mode


def mechanism_inert() -> bool:
    """Return whether the causal DAG/mechanism machinery is ablated."""
    return _mechanism_mode == "inert"


def learning_frozen() -> bool:
    """Return whether online learning is ablated."""
    return _learning_mode == "frozen"


def passthrough_mechanism_id() -> str:
    """Return the sentinel mechanism id every rank commits to in inert mode."""
    if _passthrough_mechanism_id is None:
        raise RuntimeError(
            "mechanism mode is inert but no pass-through mechanism is registered; "
            "the runner must call ensure_passthrough_mechanism at boot"
        )
    return _passthrough_mechanism_id


def ablation_status() -> dict:
    """Return the effective ablation configuration for the run manifest."""
    return {
        "mechanism_mode": _mechanism_mode,
        "learning_mode": _learning_mode,
        "passthrough_mechanism_id": _passthrough_mechanism_id,
    }
