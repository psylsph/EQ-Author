"""Progress detection and context loading utilities."""

from pathlib import Path
from typing import Dict, Any, List


def detect_progress(out_dir: Path) -> Dict[str, Any]:
    """Detect current progress from an output directory.

    Returns a dict with:
    - last_step: highest completed step (1-5), or 0 if only step 1 exists
    - last_chapter: highest completed chapter number, or 0 if no chapters
    - n_chapters: total number of chapters (from step 4 if available)
    - has_all_steps: whether all steps 1-5 are complete
    """
    from .constants import STEP_FILENAMES

    result = {"last_step": 0, "last_chapter": 0, "n_chapters": None, "has_all_steps": False}

    max_step = 0
    for step_num in range(1, 6):
        step_file = out_dir / STEP_FILENAMES.get(step_num, f"step_{step_num:02d}.md")
        if step_file.exists():
            max_step = step_num

    result["last_step"] = max_step
    result["has_all_steps"] = max_step >= 5

    if max_step >= 4:
        step4_file = out_dir / STEP_FILENAMES[4]
        try:
            step4_content = step4_file.read_text(encoding="utf-8")
            for line in step4_content.splitlines():
                if "chapter" in line.lower() and ":" in line:
                    parts = line.split(":")
                    try:
                        num = int(parts[-1].strip())
                        result["n_chapters"] = num
                        break
                    except (ValueError, IndexError):
                        continue
        except Exception:
            pass

    chapters_dir = out_dir / "chapters"
    if chapters_dir.exists():
        max_chapter = 0
        for f in chapters_dir.iterdir():
            if f.name.startswith("chapter_") and f.name.endswith(".md"):
                try:
                    num = int(f.name[8:10])
                    max_chapter = max(max_chapter, num)
                except (ValueError, IndexError):
                    continue
        result["last_chapter"] = max_chapter

    return result


def load_existing_context(out_dir: Path, progress: Dict[str, Any]) -> List[Dict[str, str]]:
    """Load existing conversation context from previous outputs.

    Returns a list of messages ready to continue the conversation.
    """
    from .constants import STEP_FILENAMES

    messages: List[Dict[str, str]] = []
    messages.append({"role": "system", "content": "You are a helpful assistant"})

    for step_num in range(1, 6):
        step_file = out_dir / STEP_FILENAMES.get(step_num, f"step_{step_num:02d}.md")
        if step_file.exists():
            try:
                content = step_file.read_text(encoding="utf-8")
                messages.append({"role": "assistant", "content": content})
            except Exception:
                pass

    last_chapter = progress.get("last_chapter", 0)
    for i in range(1, last_chapter + 1):
        chapter_file = out_dir / "chapters" / f"chapter_{i:02d}.md"
        if chapter_file.exists():
            try:
                content = chapter_file.read_text(encoding="utf-8")
                messages.append({"role": "assistant", "content": content})
            except Exception:
                pass

    return messages
