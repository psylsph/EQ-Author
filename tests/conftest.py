"""Pytest configuration and fixtures for EQ-Author tests."""

import os
import tempfile
from pathlib import Path
from typing import Generator
import pytest

# Test fixtures
@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def sample_story_file(temp_dir: Path) -> Path:
    """Create a sample story file for testing."""
    content = """# Story Idea: The Lost Library

A librarian discovers a hidden library that contains books about the future.

## Target Audience: General

The story should be engaging for adult readers who enjoy mystery and fantasy.
"""
    path = temp_dir / "story.txt"
    path.write_text(content)
    return path


@pytest.fixture
def adult_story_file(temp_dir: Path) -> Path:
    """Create a sample adult-targeted story file for testing."""
    content = """# Story Idea: Adult Romance

A passionate romance set in Paris.

## Target Audience: Adults

Contains mature themes and content.
"""
    path = temp_dir / "adult_story.txt"
    path.write_text(content)
    return path


@pytest.fixture
def mock_chapter_text() -> str:
    """Provide mock chapter text for testing."""
    return """# Chapter 1: The Beginning

The morning sun cast long shadows across the ancient library floor. Sarah walked through the towering shelves, her footsteps echoing in the silence.

She had been searching for three days when she finally found it—a door that shouldn't exist, hidden behind a bookshelf of forgotten manuscripts.

The door was old, its wood worn smooth by countless hands over centuries. A faint glow emanated from the cracks around its frame.

Sarah hesitated. She had read the legends, heard the stories passed down through generations of librarians. But none of them had prepared her for this moment.

With trembling fingers, she reached out and touched the cold metal of the handle.

The door swung open with a whisper, revealing a vast chamber filled with books that seemed to glow with an inner light. Each spine was inscribed with a date—some from the past, but many from the future.

Sarah gasped. She had found the Lost Library.

---

This chapter contains approximately 850 words of narrative prose that tells a compelling story about discovery and mystery.
"""


@pytest.fixture
def mock_step_content() -> str:
    """Provide mock step content for testing."""
    return """# Brainstorming

The story concept centers around Sarah, a young librarian who discovers a magical library containing books about the future. Key themes include:

- The nature of knowledge and destiny
- The responsibility that comes with knowing the future
- The tension between curiosity and caution

## Reflection

This concept works well because:
1. It has a strong protagonist with clear motivations
2. The magical element (future books) creates natural tension
3. The setting (library) grounds the fantastical elements

CHAPTER_COUNT: 5
"""


@pytest.fixture
def mock_multi_chapter_text() -> str:
    """Provide mock multi-chapter context text."""
    return """# Chapter 1: The Discovery

Sarah stood before the hidden door, her heart pounding.

# Chapter 2: The First Book

She reached for a book titled "2024: The Year Everything Changed."

# Chapter 3: The Warning

But the book slipped from her fingers, falling open to a page that showed her own face—in danger.
"""
