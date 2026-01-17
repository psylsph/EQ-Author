"""Tests for resume and progress detection functions."""

import pytest
from pathlib import Path
from eq_author import detect_progress, load_existing_context


class TestDetectProgress:
    """Tests for detect_progress function."""

    def test_empty_directory(self, temp_dir: Path):
        """Test detecting progress in empty directory."""
        progress = detect_progress(temp_dir)
        assert progress["last_step"] == 0
        assert progress["last_chapter"] == 0
        assert progress["n_chapters"] is None

    def test_with_step_files(self, temp_dir: Path):
        """Test detecting progress with step files."""
        # Create step files
        (temp_dir / "01_brainstorm_and_reflection.md").write_text("Step 1")
        (temp_dir / "02_intention_and_chapter_planning.md").write_text("Step 2")
        (temp_dir / "03_human_vs_llm_critique.md").write_text("Step 3")

        progress = detect_progress(temp_dir)
        assert progress["last_step"] == 3
        assert progress["has_all_steps"] is False

    def test_with_all_steps(self, temp_dir: Path):
        """Test detecting when all steps are complete."""
        for i in range(1, 6):
            (temp_dir / f"0{i}_brainstorm_and_reflection.md".replace("0"+str(i), f"0{i}" if i < 10 else str(i))).write_text(f"Step {i}")

        # Create proper step filenames
        from eq_author import STEP_FILENAMES
        for step_num, filename in STEP_FILENAMES.items():
            (temp_dir / filename).write_text(f"Step {step_num} content")

        progress = detect_progress(temp_dir)
        assert progress["last_step"] == 5
        assert progress["has_all_steps"] is True

    def test_with_chapters(self, temp_dir: Path):
        """Test detecting progress with chapter files."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir()

        # Create chapter files
        (chapters_dir / "chapter_01.md").write_text("Chapter 1")
        (chapters_dir / "chapter_02.md").write_text("Chapter 2")
        (chapters_dir / "chapter_05.md").write_text("Chapter 5")

        progress = detect_progress(temp_dir)
        assert progress["last_chapter"] == 5

    def test_extracts_chapter_count(self, temp_dir: Path):
        """Test extracting chapter count from step 4."""
        (temp_dir / "04_final_plan.md").write_text(
            "# Final Plan\nThis story will have 12 chapters.\nCHAPTER_COUNT: 12"
        )

        progress = detect_progress(temp_dir)
        assert progress["n_chapters"] == 12


class TestLoadExistingContext:
    """Tests for load_existing_context function."""

    def test_empty_directory(self, temp_dir: Path):
        """Test loading context from empty directory."""
        progress = detect_progress(temp_dir)
        messages = load_existing_context(temp_dir, progress)
        # Should have at least system message
        assert len(messages) >= 1
        assert messages[0]["role"] == "system"

    def test_loads_step_files(self, temp_dir: Path):
        """Test that step files are loaded as assistant messages."""
        from eq_author import STEP_FILENAMES
        for step_num, filename in STEP_FILENAMES.items():
            (temp_dir / filename).write_text(f"Content for step {step_num}")

        progress = detect_progress(temp_dir)
        messages = load_existing_context(temp_dir, progress)

        # Should have system message + 5 step messages
        assert len(messages) == 6
        # Steps should be loaded as assistant messages
        for i in range(1, 6):
            assert messages[i]["role"] == "assistant"
            assert f"step {i}" in messages[i]["content"].lower() or f"step {i}" in messages[i]["content"]

    def test_loads_chapters(self, temp_dir: Path):
        """Test that chapter files are loaded."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir()

        (chapters_dir / "chapter_01.md").write_text("Chapter 1 content")
        (chapters_dir / "chapter_02.md").write_text("Chapter 2 content")

        progress = detect_progress(temp_dir)
        messages = load_existing_context(temp_dir, progress)

        # Should have system + steps (0) + chapters (2)
        assert len(messages) >= 3

        # Find chapter messages
        chapter_messages = [m for m in messages if "Chapter 1 content" in m["content"] or "Chapter 2 content" in m["content"]]
        assert len(chapter_messages) == 2
