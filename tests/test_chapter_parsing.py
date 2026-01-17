"""Tests for chapter proposal parsing functions."""

import pytest
from eq_author import parse_proposed_chapters


class TestParseProposedChapters:
    """Tests for parse_proposed_chapters function."""

    def test_empty_string(self):
        """Test parsing empty string returns None."""
        assert parse_proposed_chapters("") is None

    def test_no_chapter_count(self):
        """Test text without chapter count returns None."""
        text = "This is a story about a hero."
        assert parse_proposed_chapters(text) is None

    def test_simple_chapter_count(self):
        """Test parsing simple chapter count."""
        text = "Some text\nCHAPTER_COUNT: 5"
        assert parse_proposed_chapters(text) == 5

    def test_chapter_count_with_text(self):
        """Test parsing chapter count with surrounding text."""
        text = """# Reflection
The story should have 10 chapters.
CHAPTER_COUNT: 10
Some more text"""
        assert parse_proposed_chapters(text) == 10

    def test_chapter_count_case_insensitive(self):
        """Test parsing is case insensitive."""
        assert parse_proposed_chapters("chapter_count: 7") == 7
        assert parse_proposed_chapters("Chapter_Count: 7") == 7

    def test_chapter_count_with_bold(self):
        """Test parsing chapter count with markdown bold."""
        text = "**CHAPTER_COUNT:** 12"
        assert parse_proposed_chapters(text) == 12

    def test_chapter_count_not_at_start(self):
        """Test parsing finds count even not at start."""
        text = "Some text before\nCHAPTER_COUNT: 3\nSome text after"
        assert parse_proposed_chapters(text) == 3

    def test_invalid_chapter_count(self):
        """Test invalid chapter count returns None."""
        text = "CHAPTER_COUNT: not_a_number"
        assert parse_proposed_chapters(text) is None

    def test_zero_chapters(self):
        """Test zero chapter count is valid."""
        assert parse_proposed_chapters("CHAPTER_COUNT: 0") == 0

    def test_negative_chapters(self):
        """Test negative chapter count is parsed (validation happens elsewhere)."""
        assert parse_proposed_chapters("CHAPTER_COUNT: -1") == -1

    def test_large_chapter_count(self):
        """Test large chapter count is parsed correctly."""
        assert parse_proposed_chapters("CHAPTER_COUNT: 100") == 100

    def test_chapter_count_with_spaces(self):
        """Test chapter count with extra spaces."""
        assert parse_proposed_chapters("CHAPTER_COUNT:   6   ") == 6

    def test_searches_reversed(self):
        """Test that search is in reversed order (finds last match)."""
        text = """CHAPTER_COUNT: 3
More text
CHAPTER_COUNT: 5
Final text"""
        assert parse_proposed_chapters(text) == 5

