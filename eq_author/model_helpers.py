"""Model-specific helper functions."""

from .constants import (
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEEPSEEK_REASONER_DEFAULT_TOKENS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEEPSEEK_MAX_OUTPUT_TOKENS,
    DEEPSEEK_REASONER_MAX_OUTPUT_TOKENS,
    DEFAULT_TEMPERATURE,
    DEEPSEEK_REASONER_TEMPERATURE,
)


def get_default_max_context_tokens(model: str) -> int:
    """Return appropriate default max_context_tokens based on model."""
    if model.lower() == "deepseek-reasoner":
        return DEEPSEEK_REASONER_DEFAULT_TOKENS
    return DEFAULT_MAX_CONTEXT_TOKENS


def get_default_max_output_tokens(model: str) -> int:
    """Return appropriate default max_output_tokens based on model."""
    model_lower = model.lower()
    if "deepseek-reasoner" in model_lower:
        return DEEPSEEK_REASONER_MAX_OUTPUT_TOKENS
    if "deepseek" in model_lower:
        return DEEPSEEK_MAX_OUTPUT_TOKENS
    return DEFAULT_MAX_OUTPUT_TOKENS


def calculate_safe_max_tokens(context_tokens: int, max_context: int, model: str) -> int:
    """Calculate a safe max_tokens value that leaves room for response."""
    default_output = get_default_max_output_tokens(model)
    available = max_context - context_tokens
    reserved = int(available * 0.20)
    return min(default_output, max(reserved, 500))


def get_default_temperature(model: str) -> float:
    """Return appropriate default temperature based on model."""
    if model.lower() == "deepseek-reasoner":
        return DEEPSEEK_REASONER_TEMPERATURE
    return DEFAULT_TEMPERATURE
