"""Main CLI entry point for EQ-Author."""

import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from .cli import parse_args
from .api import make_client
from .workflow_v2 import run_workflow_v2
from .workflow_skip_planning import run_workflow_v2_skip_planning
from .workflow_resume import run_workflow_v2_resume
from .model_helpers import get_default_temperature
from typing import Optional, Callable


def _get_api_key() -> str:
    """Get API key from environment variables."""
    import os
    return os.getenv("EQ_AUTHOR_API_KEY") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or ""


def _get_cache(no_cache: bool, cache_dir: str) -> object:
    """Initialize prompt cache if enabled."""
    from .api import PromptCache
    if no_cache:
        return None
    if not cache_dir:
        return None
    try:
        return PromptCache(Path(cache_dir))
    except Exception as exc:
        print(f"Warning: prompt cache disabled ({exc})")
        return None


def main(argv: List[str]) -> int:


def main(argv: List[str]) -> int:
    """Main entry point."""
    load_dotenv()

    args = parse_args(argv)

    if args.resume_from:
        resume_dir = Path(args.resume_from)
        if not resume_dir.exists():
            print(f"Error: Resume directory not found: {resume_dir}", file=sys.stderr)
            return 2

        api_key = args.api_key or _get_api_key()
        if not api_key:
            print("Error: Provide --api-key or set API_KEY/OPENAI_API_KEY", file=sys.stderr)
            return 2

        cache = _get_cache(args.no_cache, args.cache_dir)

        stream_enabled = args.stream and not args.no_stream
        temperature = args.temperature if args.temperature is not None else get_default_temperature(args.model)

        try:
            run_workflow_v2_resume(
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                story_prompt="",
                out_dir=resume_dir,
                stream=stream_enabled,
                temperature=temperature,
                cache=cache,
                context_strategy=args.context_strategy,
                summary_length=args.summary_length,
                recent_chapters=args.recent_chapters,
                max_context_tokens=args.max_context_tokens,
                always_autogen_chapters=args.always_autogen_chapters,
                n_chapters=args.n_chapters,
            )
        except Exception as e:
            print(f"Error during resume: {e}", file=sys.stderr)
            return 1

        print("Resume complete.")
        return 0

    api_key = args.api_key or _get_api_key()
    if not api_key:
        print("Error: Provide --api-key or set API_KEY/OPENAI_API_KEY", file=sys.stderr)
        return 2

    try:
        from .file_io import load_story_prompt, ensure_output_dir
    except Exception:
        pass

    try:
        story_prompt = load_story_prompt(args.story_file, args.story_text)
    except Exception as e:
        print(f"Error loading story prompt: {e}", file=sys.stderr)
        return 2

    out_dir = ensure_output_dir(args.output_dir)
    print(f"Writing outputs to: {out_dir}")

    cache = _get_cache(args.no_cache, args.cache_dir)

    stream_enabled = args.stream and not args.no_stream
    temperature = args.temperature if args.temperature is not None else get_default_temperature(args.model)

    try:
        if args.skip_planning:
            from .workflow_skip_planning import run_workflow_v2_skip_planning
            run_workflow_v2_skip_planning(
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                out_dir=out_dir,
                stream=stream_enabled,
                temperature=temperature,
                cache=cache,
                feedback_callback=None,
                stream_callback=None,
                progress_callback=None,
                context_strategy=args.context_strategy,
                summary_length=args.summary_length,
                recent_chapters=args.recent_chapters,
                max_context_tokens=args.max_context_tokens,
                always_autogen_chapters=args.always_autogen_chapters,
            )
        else:
            run_workflow_v2(
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                story_prompt=story_prompt,
                n_chapters=args.n_chapters,
                interactive=not args.non_interactive,
                out_dir=out_dir,
                stream=stream_enabled,
                temperature=temperature,
                cache=cache,
                feedback_callback=None,
                stream_callback=None,
                progress_callback=None,
                context_strategy=args.context_strategy,
                summary_length=args.summary_length,
                recent_chapters=args.recent_chapters,
                max_context_tokens=args.max_context_tokens,
                always_autogen_chapters=args.always_autogen_chapters,
            )
    except Exception as e:
        print(f"Error during workflow: {e}", file=sys.stderr)
        return 1

    print("Done.")
    return 0


def _get_api_key() -> str:
    """Get API key from environment variables."""
    import os
    return os.getenv("EQ_AUTHOR_API_KEY") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or ""


def _get_cache(no_cache: bool, cache_dir: str) -> object:
    """Initialize prompt cache if enabled."""
    if no_cache:
        return None
    if not cache_dir:
        return None
    try:
        return PromptCache(Path(cache_dir))
    except Exception as exc:
        print(f"Warning: prompt cache disabled ({exc})")
        return None


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
