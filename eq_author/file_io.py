"""File I/O utilities for reading and writing story content."""

import re
from pathlib import Path
from typing import Optional


def load_story_prompt(story_file: Optional[str], story_text: Optional[str]) -> str:
    """Load story prompt from file or direct text."""
    if story_text and story_text.strip():
        return story_text.strip()
    if not story_file:
        raise ValueError("Provide either --story-file or --story-text")
    p = Path(story_file)
    if not p.exists():
        raise FileNotFoundError(f"Story file not found: {p}")
    return p.read_text(encoding="utf-8").strip()


def is_adult_target(story_text: str) -> bool:
    """Return True if story text/metadata indicates adult target audience."""
    if not story_text:
        return False
    lower = story_text.lower()

    for line in story_text.splitlines():
        l = line.lower().strip()
        if "target audience" in l and ("adult" in l or "adults" in l or "18+" in l):
            return True
        if l.startswith("**target audience:**") and ("adult" in l or "adults" in l or "18+" in l):
            return True

    head = "\n".join(story_text.splitlines()[:10]).lower()
    if "adult" in head or "erotic" in head:
        return True

    return False


def load_prompt_override_file() -> str:
    """Load project-level prompt_override.txt if present."""
    try:
        base = Path(__file__).parent.parent
        p = base / "prompt_override.txt"
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return ""


def strip_injection_markers(text: str) -> str:
    """Remove injection markers like [^420] from text."""
    lines = text.splitlines(keepends=True)
    filtered = [line for line in lines if not line.strip().startswith("[^")]
    result = "".join(filtered)
    result = re.sub(r'\[\^[0-9]+\]', '', result)
    return result.strip()


def strip_thinking_sections(text: str) -> str:
    """Remove thinking/planning sections from LLM output."""
    if not text:
        return text

    lines = text.splitlines()
    heading_patterns = [r'^#\s+', r'^##\s+', r'^Chapter\s+\d+', r'^Part\s+\d+']
    content_start_index = 0
    found_content = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        for pattern in heading_patterns:
            if re.match(pattern, stripped, re.IGNORECASE):
                content_start_index = i
                found_content = True
                break
        if found_content:
            break

    if found_content:
        return "\n".join(lines[content_start_index:]).strip()
    return text


def parse_proposed_chapters(text: str) -> Optional[int]:
    """Extract chapter count from step 1 output."""
    for line in reversed(text.splitlines()):
        line = line.strip().replace("*", "")
        if line.upper().find("CHAPTER_COUNT:") >= 0:
            parts = line.split(":", 1)
            if len(parts) == 2:
                num = parts[1].strip()
                try:
                    return int(num)
                except ValueError:
                    return None
    return None


def collect_feedback(step_label: str) -> str:
    """Collect user feedback for a step."""
    try:
        print(f"\nProvide feedback for {step_label} (press Enter to skip):")
        fb = input().strip()
        return fb
    except KeyboardInterrupt:
        print("\nSkipping feedback.")
        return ""


def ensure_output_dir(base_dir: str) -> Path:
    """Create output directory with timestamp."""
    from datetime import datetime
    base = Path(base_dir)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = base / f"run-{ts}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "chapters").mkdir(parents=True, exist_ok=True)
    return out


def write_step_output(out_dir: Path, step: int, content: str) -> Path:
    """Write step output to file."""
    from .constants import STEP_FILENAMES

    name = STEP_FILENAMES.get(step, f"step_{step:02d}.md")
    p = out_dir / name
    content = strip_injection_markers(content)
    content = strip_thinking_sections(content)
    p.write_text(content, encoding="utf-8")
    return p


def write_chapter_output(out_dir: Path, chapter_index: int, content: str) -> Path:
    """Write chapter output to file."""
    p = out_dir / "chapters" / f"chapter_{chapter_index:02d}.md"
    content = strip_injection_markers(content)
    content = strip_thinking_sections(content)
    p.write_text(content, encoding="utf-8")
    return p
