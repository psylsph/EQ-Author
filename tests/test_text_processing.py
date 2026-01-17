"""Tests for word counting and text processing utilities."""

import pytest
from eq_author import count_words, extract_chapter_ending, strip_injection_markers, strip_thinking_sections, validate_chapter_output


class TestCountWords:
    """Tests for count_words function."""

    def test_empty_string(self):
        """Test counting words in empty string."""
        assert count_words("") == 0

    def test_single_word(self):
        """Test counting single word."""
        assert count_words("hello") == 1

    def test_multiple_words(self):
        """Test counting multiple words."""
        text = "The quick brown fox jumps over the lazy dog"
        assert count_words(text) == 9

    def test_words_with_apostrophes(self):
        """Test counting words with apostrophes."""
        text = "It's a beautiful day"
        assert count_words(text) == 4

    def test_words_with_hyphens(self):
        """Test counting hyphenated words."""
        text = "well-known state-of-the-art design"
        assert count_words(text) == 3

    def test_special_characters_ignored(self):
        """Test that special characters are not counted as words."""
        text = "Hello, world! How are you?"
        assert count_words(text) == 5

    def test_newlines_and_spaces(self):
        """Test handling of newlines and multiple spaces."""
        text = "Hello\n\nWorld   Test"
        assert count_words(text) == 3


class TestExtractChapterEnding:
    """Tests for extract_chapter_ending function."""

    def test_empty_string(self):
        """Test extracting ending from empty string."""
        assert extract_chapter_ending("") == ""

    def test_single_sentence(self):
        """Test with single sentence."""
        text = "The story ends here."
        assert extract_chapter_ending(text) == "The story ends here."

    def test_two_sentences(self):
        """Test with two sentences - returns both."""
        text = "First sentence. Second sentence."
        result = extract_chapter_ending(text)
        assert "Second sentence" in result

    def test_multiple_sentences(self):
        """Test with multiple sentences - returns last two."""
        text = "First. Second. Third. Fourth."
        result = extract_chapter_ending(text)
        assert "Third" in result
        assert "Fourth" in result

    def test_with_paragraphs(self):
        """Test with paragraph breaks."""
        text = "First paragraph.\n\nSecond paragraph with more text.\n\nFinal sentence."
        result = extract_chapter_ending(text)
        assert "Final sentence" in result

    def test_exclamation_marks(self):
        """Test with exclamation marks as sentence endings."""
        text = "What a surprise! She couldn't believe it!"
        result = extract_chapter_ending(text)
        assert "believe it!" in result


class TestStripInjectionMarkers:
    """Tests for strip_injection_markers function."""

    def test_empty_string(self):
        """Test stripping from empty string."""
        assert strip_injection_markers("") == ""

    def test_no_markers(self):
        """Test text without injection markers."""
        text = "This is normal text without any markers."
        assert strip_injection_markers(text) == text

    def test_single_marker(self):
        """Test removing single injection marker."""
        text = "Some text [^420] more text"
        result = strip_injection_markers(text)
        assert "[^420]" not in result
        assert "Some text  more text" in result

    def test_multiple_markers(self):
        """Test removing multiple injection markers."""
        text = "Text [^420] more [^420] content [^420]"
        result = strip_injection_markers(text)
        assert "[^420]" not in result

    def test_marker_at_start(self):
        """Test removing marker at start of line."""
        text = "[^420] Beginning marker"
        result = strip_injection_markers(text)
        assert "[^420]" not in result

    def test_marker_at_end(self):
        """Test removing marker at end of line."""
        text = "Ending marker [^420]"
        result = strip_injection_markers(text)
        assert "[^420]" not in result

    def test_preserves_other_content(self):
        """Test that non-marker content is preserved."""
        text = "Important [^420] content"
        result = strip_injection_markers(text)
        # Markers are removed, may have double space
        assert "[^420]" not in result
        assert "Important" in result
        assert "content" in result


class TestStripThinkingSections:
    """Tests for strip_thinking_sections function."""

    def test_empty_string(self):
        """Test stripping from empty string."""
        assert strip_thinking_sections("") == ""

    def test_no_thinking_section(self):
        """Test text without thinking section."""
        text = "# Chapter 1\nThe story begins here."
        assert strip_thinking_sections(text) == text

    def test_thinking_then_content(self):
        """Test removing thinking section before content."""
        text = """Let me think about this carefully. The story should have tension.

# Chapter 1
The story begins here."""
        result = strip_thinking_sections(text)
        assert result.startswith("# Chapter 1")

    def test_chapter_heading(self):
        """Test finding chapter heading."""
        text = "Planning thoughts here.\n\nChapter 1: The Beginning\nStory content."
        result = strip_thinking_sections(text)
        assert "Chapter 1: The Beginning" in result

    def test_part_heading(self):
        """Test finding part heading."""
        text = "Thinking about structure.\n\nPart 1: The Journey\nStory content."
        result = strip_thinking_sections(text)
        assert "Part 1: The Journey" in result

    def test_only_heading(self):
        """Test with only heading."""
        text = "# Chapter 5"
        assert strip_thinking_sections(text) == "# Chapter 5"

    def test_ignores_empty_lines_at_start(self):
        """Test that empty lines at start are skipped."""
        text = "\n\n\n# Chapter 1\nContent"
        result = strip_thinking_sections(text)
        assert result.startswith("# Chapter 1")

    def test_case_insensitive_heading(self):
        """Test heading detection is case insensitive."""
        text = "Thinking...\n\nCHAPTER 5: The End\nContent"
        result = strip_thinking_sections(text)
        assert "CHAPTER 5" in result


class TestValidateChapterOutput:
    """Tests for validate_chapter_output function."""

    def test_valid_chapter(self):
        """Test that a valid chapter passes validation."""
        text = """# Chapter 1

The story begins with a young adventurer setting out on a journey. She had prepared for this moment her entire life, training her mind and body for the challenges ahead.

The ancient ruins stood before her, their stone walls covered in moss and time. She took a deep breath and stepped forward into the unknown. The door was heavy, but it opened with a whispered creak.

This was it - the moment she had been searching for her entire life. She knew that whatever lay beyond would change everything."""
        is_valid, issue = validate_chapter_output(text)
        assert is_valid is True
        assert issue == ""

    def test_chapter_ends_mid_sentence(self):
        """Test that a chapter ending mid-sentence fails validation."""
        text = """# Chapter 1

The story begins with a young adventurer. She had prepared for this moment her entire life

The ancient ruins stood before her"""
        is_valid, issue = validate_chapter_output(text)
        assert is_valid is False
        assert "mid-sentence" in issue.lower()

    def test_repetitive_sentence(self):
        """Test that repetitive sentences fail validation."""
        text = """# Chapter 1

The story begins with a young adventurer setting out on a journey. The story begins with a young adventurer setting out on a journey. The story begins with a young adventurer setting out on a journey. The story begins with a young adventurer setting out on a journey. The story begins with a young adventurer setting out on a journey.

The ancient ruins stood before her, their stone walls covered in moss and time. She had prepared for this moment her entire life."""
        is_valid, issue = validate_chapter_output(text)
        assert is_valid is False
        assert "repetitive" in issue.lower()

    def test_repetitive_lines(self):
        """Test that repetitive lines fail validation."""
        text = """# Chapter 1

He gave her a chance to prove herself.
He gave her a chance to prove herself.
He gave her a chance to prove herself.

The ancient ruins stood before her, their stone walls covered in moss and time. She had prepared for this moment her entire life."""
        is_valid, issue = validate_chapter_output(text)
        assert is_valid is False
        assert "repetitive" in issue.lower()

    def test_character_repetition(self):
        """Test that excessive character repetition fails validation."""
        text = """# Chapter 1

The story begins with aaaaaaaa great adventure. The ancient ruins stood before her.

She had trained for this moment."""
        is_valid, issue = validate_chapter_output(text)
        assert is_valid is False
        assert "character repetition" in issue.lower()

    def test_too_short(self):
        """Test that too-short text fails validation."""
        text = """# Chapter 1

Short."""
        is_valid, issue = validate_chapter_output(text)
        assert is_valid is False
        assert "too short" in issue.lower()

    def test_empty_string(self):
        """Test that empty string fails validation."""
        is_valid, issue = validate_chapter_output("")
        assert is_valid is False

    def test_valid_with_dialogue(self):
        """Test that valid chapter with dialogue passes."""
        text = """# Chapter 1

"Are you sure about this?" Sarah asked.

"No," John replied. "But we have no choice."

They pressed forward into the darkness, their flashlights cutting narrow beams through the gloom."""
        is_valid, issue = validate_chapter_output(text)
        assert is_valid is True
