"""Workflow for generating chapters using existing planning files."""

from pathlib import Path
from typing import Optional, Callable

from .api import make_client, PromptCache
from .context import ContextManager
from .progress import detect_progress, load_existing_context
from .file_io import write_chapter_output, collect_feedback
from .prompts import chapter_prompt
from .constants import DEFAULT_SUMMARY_LENGTH
from .workflow_utils import (
    _setup_callbacks,
    _generate_chapter_auto,
    _generate_chapter_interactive,
    _auto_review_chapter,
    _apply_feedback_rewrite,
)
from .model_helpers import calculate_safe_max_tokens, get_default_max_context_tokens
from .chapter import generate_chapter_summary
from .text_utils import extract_chapter_ending


def run_workflow_v2_skip_planning(
    api_key: str,
    base_url: str,
    model: str,
    out_dir: Path,
    stream: bool = False,
    temperature: Optional[float] = None,
    cache: Optional[PromptCache] = None,
    feedback_callback: Optional[Callable[[str], str]] = None,
    stream_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    context_strategy: str = "aggressive",
    summary_length: int = DEFAULT_SUMMARY_LENGTH,
    recent_chapters: int = 2,
    max_context_tokens: Optional[int] = None,
    always_autogen_chapters: bool = False,
) -> None:
    """Generate chapters using existing planning files (steps 1-5 must exist in out_dir).

    This function skips the planning workflow and uses existing planning files
    to generate chapters directly. Useful for when you've prepared planning
    files manually or want to regenerate chapters with different settings.
    """
    client = make_client(api_key, base_url)
    token_handler, emit_progress, maybe_collect_feedback = _setup_callbacks(stream, stream_callback, progress_callback)

    progress = detect_progress(out_dir)
    last_step = progress["last_step"]
    last_chapter = progress["last_chapter"]

    if last_step < 5:
        raise RuntimeError(
            f"Planning files incomplete. Found steps 1-{last_step}, but need all steps 1-5. "
            f"Use --resume-from with a directory containing complete planning files, "
            f"or run without --skip-planning to generate planning files first."
        )

    print(f"Using existing planning files from {out_dir}")
    print(f"Detected {last_chapter} existing chapters")

    messages = load_existing_context(out_dir, progress)
    print(f"Loaded {len(messages)} messages from planning files")

    if max_context_tokens is None:
        max_context_tokens = get_default_max_context_tokens(model)

    context_manager = ContextManager(
        strategy=context_strategy,
        summary_length=summary_length,
        recent_chapters=recent_chapters,
        max_context_tokens=max_context_tokens,
    )

    context_manager.add_core_context(messages)

    if last_chapter > 0:
        for i in range(1, last_chapter + 1):
            chapter_file = out_dir / "chapters" / f"chapter_{i:02d}.md"
            summary_file = out_dir / "chapters" / f"chapter_{i:02d}_summary.md"

            if chapter_file.exists():
                try:
                    chapter_text = chapter_file.read_text(encoding="utf-8")
                    chapter_ending = extract_chapter_ending(chapter_text)

                    if summary_file.exists():
                        chapter_summary = summary_file.read_text(encoding="utf-8")
                    else:
                        chapter_summary = chapter_text

                    context_manager.add_chapter(i, chapter_text, chapter_summary, chapter_ending)
                except Exception as e:
                    print(f"Warning: Failed to load chapter {i}: {e}")

    n_chapters = progress.get("n_chapters")
    if n_chapters is None:
        from .file_io import parse_proposed_chapters
        step4_file = out_dir / "04_final_plan.md"
        if step4_file.exists():
            step4_content = step4_file.read_text(encoding="utf-8")
            for line in step4_content.splitlines():
                if "chapter" in line.lower() and ":" in line:
                    parts = line.split(":")
                    try:
                        n_chapters = int(parts[-1].strip())
                        break
                    except (ValueError, IndexError):
                        continue
        if n_chapters is None:
            raise RuntimeError("Could not determine chapter count from planning files. Ensure step 4 contains a chapter count (e.g., 'CHAPTER_COUNT: 10').")

    print(f"Generating chapters {last_chapter + 1} through {n_chapters}")

    for i in range(last_chapter + 1, n_chapters + 1):
        previous_chapter_ending = context_manager.get_previous_chapter_ending(i) if i > 1 else ""
        context_messages = context_manager.build_context(i)
        context_messages.append({"role": "user", "content": chapter_prompt(i, previous_chapter_ending)})

        context_status = context_manager.check_context_size(context_messages)
        if context_status["is_warning"]:
            print(f"Warning: Context usage at {context_status['usage_ratio']:.1%} ({context_status['estimated_tokens']}/{context_status['max_tokens']} tokens)", flush=True)
        if context_status["is_critical"]:
            print(f"CRITICAL: Context usage at {context_status['usage_ratio']:.1%} - consider reducing context size", flush=True)

        if stream:
            print(f"\n=== Chapter {i} (streaming) ===", flush=True)
        emit_progress(f"Chapter {i}")

        if always_autogen_chapters:
            final_text, last_word_count = _generate_chapter_auto(
                client, model, i, context_manager, context_messages, stream, temperature, cache, token_handler, emit_progress
            )
        else:
            final_text, last_word_count = _generate_chapter_interactive(
                client, model, i, context_manager, context_messages, stream, temperature, cache, token_handler, emit_progress, maybe_collect_feedback
            )

        if always_autogen_chapters:
            final_text, last_word_count = _auto_review_chapter(
                client, model, i, final_text, context_manager, out_dir, stream, temperature, cache
            )

        write_chapter_output(out_dir, i, final_text)

        chapter_ending = extract_chapter_ending(final_text)

        if stream:
            print(f"\n=== Generating Chapter {i} Summary ===", flush=True)
        chapter_summary = generate_chapter_summary(client, model, final_text, i, summary_length, stream, temperature, cache)

        summary_file = out_dir / "chapters" / f"chapter_{i:02d}_summary.md"
        summary_file.write_text(chapter_summary, encoding="utf-8")

        context_manager.add_chapter(i, final_text, chapter_summary, chapter_ending)

        if not always_autogen_chapters:
            fb = maybe_collect_feedback(f"chapter {i}")
            if fb:
                context_messages.append({"role": "user", "content": f"FEEDBACK AFTER CHAPTER {i}:\n{fb}\nPlease apply this feedback in the next chapter(s)."})
