"""CLI argument parsing."""

import argparse
import os
from typing import Optional


def get_env_var_for_arg(arg_name: str) -> Optional[str]:
    """Convert argument name to env var name and return its value."""
    env_var = f"EQ_AUTHOR_{arg_name.upper().replace('-', '_')}"
    return os.getenv(env_var)


def parse_args(argv):
    """Parse command line arguments."""
    from .constants import DEFAULT_CONTEXT_STRATEGY, DEFAULT_SUMMARY_LENGTH, DEFAULT_RECENT_CHAPTERS, DEFAULT_TEMPERATURE

    p = argparse.ArgumentParser(description="EQ-Author DeepSeek Planner & Writer")
    p.add_argument("--story-file", type=str, help="Path to file containing story idea", default=get_env_var_for_arg("story-file"))
    p.add_argument("--story-text", type=str, help="Story idea text (overrides file if provided)", default=get_env_var_for_arg("story-text"))
    p.add_argument("--n-chapters", type=int, help="Override: number of chapters to write (if omitted, model proposes)", default=None)
    env_n_chapters = get_env_var_for_arg("n-chapters")
    if env_n_chapters:
        try:
            p.set_defaults(n_chapters=int(env_n_chapters))
        except ValueError:
            pass
    p.add_argument("--output-dir", type=str, help="Base directory for outputs", default=get_env_var_for_arg("output-dir") or "outputs")
    p.add_argument("--api-key", type=str, help="DeepSeek API key (or set API_KEY or EQ_AUTHOR_API_KEY)", default=None)
    p.add_argument("--base-url", type=str, help="DeepSeek base URL", default=get_env_var_for_arg("base-url") or "https://api.deepseek.com")
    p.add_argument("--model", type=str, help="Model name", default=get_env_var_for_arg("model") or "deepseek-reasoner")
    p.add_argument("--stream", action="store_true", help="Stream responses (aggregated in output)", default=True)
    p.add_argument("--no-stream", action="store_true", help="Disable response streaming", default=False)
    env_no_stream = get_env_var_for_arg("no-stream")
    if env_no_stream and env_no_stream.lower() in ("true", "1", "yes"):
        p.set_defaults(no_stream=True)
    p.add_argument(
        "--temperature",
        type=float,
        help=f"Sampling temperature (default: model-specific, {DEFAULT_TEMPERATURE} for most models, 1.0 for deepseek-reasoner)",
        default=None,
    )
    env_temp = get_env_var_for_arg("temperature")
    if env_temp:
        try:
            p.set_defaults(temperature=float(env_temp))
        except ValueError:
            pass
    p.add_argument("--non-interactive", action="store_true", help="Run without feedback prompts and chapter count confirmation", default=False)
    env_non_interactive = get_env_var_for_arg("non-interactive")
    if env_non_interactive and env_non_interactive.lower() in ("true", "1", "yes"):
        p.set_defaults(non_interactive=True)
    p.add_argument("--no-cache", action="store_true", help="Disable prompt caching", default=False)
    env_no_cache = get_env_var_for_arg("no-cache")
    if env_no_cache and env_no_cache.lower() in ("true", "1", "yes"):
        p.set_defaults(no_cache=True)
    p.add_argument(
        "--cache-dir",
        type=str,
        default=get_env_var_for_arg("cache-dir") or ".prompt_cache",
        help="Directory for prompt/response cache (ignored if --no-cache is used)",
    )

    p.add_argument(
        "--context-strategy",
        type=str,
        choices=["aggressive", "balanced"],
        default=get_env_var_for_arg("context-strategy") or DEFAULT_CONTEXT_STRATEGY,
        help="Strategy for managing context window (default: aggressive)",
    )
    p.add_argument(
        "--summary-length",
        type=int,
        default=DEFAULT_SUMMARY_LENGTH,
        help=f"Target word count for chapter summaries (default: {DEFAULT_SUMMARY_LENGTH})",
    )
    env_summary = get_env_var_for_arg("summary-length")
    if env_summary:
        try:
            p.set_defaults(summary_length=int(env_summary))
        except ValueError:
            pass
    p.add_argument(
        "--recent-chapters",
        type=int,
        default=DEFAULT_RECENT_CHAPTERS,
        help=f"Number of recent full chapters to keep in context (default: {DEFAULT_RECENT_CHAPTERS})",
    )
    env_recent = get_env_var_for_arg("recent-chapters")
    if env_recent:
        try:
            p.set_defaults(recent_chapters=int(env_recent))
        except ValueError:
            pass
    p.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        help="Maximum context tokens to maintain (default: model-specific, 8000 for most models, 32000 for deepseek-reasoner)",
    )
    env_max_tokens = get_env_var_for_arg("max-context-tokens")
    if env_max_tokens:
        try:
            p.set_defaults(max_context_tokens=int(env_max_tokens))
        except ValueError:
            pass
    p.add_argument("--resume-from", type=str, default=None, help="Resume from a previous run by providing output directory path")
    p.add_argument("--skip-planning", action="store_true", default=False, help="Skip planning steps (1-5) and use existing files from --output-dir for chapter generation")
    env_skip = get_env_var_for_arg("skip-planning")
    if env_skip and env_skip.lower() in ("true", "1", "yes"):
        p.set_defaults(skip_planning=True)
    p.add_argument(
        "--always-autogen-chapters",
        action="store_true",
        default=False,
        help="After completing planning steps (1-5), automatically generate all chapters without prompting for feedback between chapters",
    )
    env_autogen = get_env_var_for_arg("always-autogen-chapters")
    if env_autogen and env_autogen.lower() in ("true", "1", "yes"):
        p.set_defaults(always_autogen_chapters=True)

    return p.parse_args(argv)
