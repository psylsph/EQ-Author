"""Tests for workflow utility functions."""

import pytest
from pathlib import Path
from eq_author import (
    build_step1_prompt,
    build_followup_prompts,
    ensure_output_dir,
    write_step_output,
    write_chapter_output,
    chapter_prompt,
    get_default_max_context_tokens,
    get_default_temperature,
)


class TestBuildPrompts:
    """Tests for prompt building functions."""

    def test_build_step1_prompt(self):
        """Test that step 1 prompt contains key elements."""
        prompt = build_step1_prompt()
        assert "Brainstorming" in prompt
        assert "Reflection" in prompt
        assert "CHAPTER_COUNT:" in prompt
        assert "{STORY_PROMPT}" in prompt  # Should be replaced later

    def test_build_followup_prompts_count(self):
        """Test that followup prompts returns 4 prompts."""
        prompts = build_followup_prompts(5)
        assert len(prompts) == 4

    def test_build_followup_prompts_contain_n_chapters(self):
        """Test that prompts contain chapter count."""
        prompts = build_followup_prompts(10)
        assert "10" in prompts[0]
        assert "10" in prompts[2]

    def test_build_followup_prompts_contain_word_target(self):
        """Test that prompts contain word target."""
        prompts = build_followup_prompts(5)
        assert "2000" in prompts[2]  # CHAPTER_WORD_TARGET


class TestEnsureOutputDir:
    """Tests for ensure_output_dir function."""

    def test_creates_output_directory(self, temp_dir: Path):
        """Test that output directory is created."""
        out_dir = ensure_output_dir(str(temp_dir))
        assert out_dir.exists()
        assert out_dir.is_dir()

    def test_creates_chapters_subdirectory(self, temp_dir: Path):
        """Test that chapters subdirectory is created."""
        out_dir = ensure_output_dir(str(temp_dir))
        chapters_dir = out_dir / "chapters"
        assert chapters_dir.exists()
        assert chapters_dir.is_dir()

    def test_timestamp_in_directory_name(self, temp_dir: Path):
        """Test that directory name contains timestamp."""
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            out_dir = ensure_output_dir(".")
            name = out_dir.name
            assert name.startswith("run-")
            # Timestamp format: YYYYMMDD-HHMMSS
            assert len(name) == len("run-YYYYMMDD-HHMMSS")
        finally:
            os.chdir(original_cwd)


class TestWriteOutput:
    """Tests for write output functions."""

    def test_write_step_output(self, temp_dir: Path):
        """Test writing step output file."""
        out_dir = ensure_output_dir(str(temp_dir))
        content = "# Step 1 Content\nTest content here."
        path = write_step_output(out_dir, 1, content)
        assert path.exists()
        assert path.read_text() == content

    def test_write_chapter_output(self, temp_dir: Path):
        """Test writing chapter output file."""
        out_dir = ensure_output_dir(str(temp_dir))
        content = "# Chapter 5\nStory content here."
        path = write_chapter_output(out_dir, 5, content)
        assert path.exists()
        assert path.name == "chapter_05.md"
        assert path.read_text() == content

    def test_step_filename_mapping(self, temp_dir: Path):
        """Test that step numbers map to correct filenames."""
        out_dir = ensure_output_dir(str(temp_dir))
        for step_num, expected_name in {
            1: "01_brainstorm_and_reflection.md",
            2: "02_intention_and_chapter_planning.md",
            3: "03_human_vs_llm_critique.md",
            4: "04_final_plan.md",
            5: "05_characters.md",
        }.items():
            write_step_output(out_dir, step_num, "test")
            assert (out_dir / expected_name).exists()


class TestChapterPrompt:
    """Tests for chapter_prompt function."""

    def test_chapter_1_prompt(self):
        """Test that chapter 1 prompt is different."""
        prompt1 = chapter_prompt(1)
        prompt2 = chapter_prompt(2)
        assert "Great. Let's begin the story." in prompt1
        assert "Continue with the next installment" in prompt2

    def test_chapter_prompt_contains_word_count(self):
        """Test that prompt contains word count requirement."""
        for i in [1, 2, 5]:
            prompt = chapter_prompt(i)
            assert "2000" in prompt  # CHAPTER_MIN_WORDS

    def test_chapter_prompt_prevents_overlap(self):
        """Test that later chapters include overlap prevention."""
        prompt = chapter_prompt(5)
        assert "previous chapter" in prompt.lower() or "previous" in prompt.lower()

    def test_chapter_prompt_with_ending(self):
        """Test that ending is included when provided."""
        prompt = chapter_prompt(3, "The door creaked open.")
        assert "door creaked open" in prompt or "previous chapter ended" in prompt.lower()


class TestModelDefaults:
    """Tests for model-specific default functions."""

    def test_deepseek_reasoner_max_context(self):
        """Test deepseek-reasoner has correct max context."""
        assert get_default_max_context_tokens("deepseek-reasoner") == 32000

    def test_other_model_max_context(self):
        """Test other models have default max context."""
        assert get_default_max_context_tokens("other-model") == 8000
        assert get_default_max_context_tokens("gpt-4") == 8000

    def test_deepseek_reasoner_temperature(self):
        """Test deepseek-reasoner has correct temperature."""
        assert get_default_temperature("deepseek-reasoner") == 1.0

    def test_other_model_temperature(self):
        """Test other models have default temperature."""
        assert get_default_temperature("other-model") == 1.0

    def test_case_insensitive_model_detection(self):
        """Test that model detection is case insensitive."""
        assert get_default_max_context_tokens("DeepSeek-Reasoner") == 32000
        assert get_default_temperature("DEEPSEEK-REASONER") == 1.0
