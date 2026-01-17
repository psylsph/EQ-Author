"""Tests for PromptCache class."""

import json
import pytest
from pathlib import Path
from eq_author import PromptCache


class TestPromptCacheInit:
    """Tests for PromptCache initialization."""

    def test_creates_directory(self, temp_dir: Path):
        """Test that cache creates directory on init."""
        cache_dir = temp_dir / ".test_cache"
        cache = PromptCache(cache_dir)
        assert cache_dir.exists()
        assert cache_dir.is_dir()

    def test_existing_directory(self, temp_dir: Path):
        """Test that cache works with existing directory."""
        cache_dir = temp_dir / ".existing_cache"
        cache_dir.mkdir()
        cache = PromptCache(cache_dir)
        assert cache_dir.exists()


class TestPromptCachePath:
    """Tests for cache path generation."""

    def test_same_inputs_same_path(self, temp_dir: Path):
        """Test that same inputs produce same cache path."""
        cache = PromptCache(temp_dir / "cache")
        messages = [{"role": "user", "content": "test"}]

        path1 = cache._cache_path(model="model1", messages=messages, temperature=0.5)
        path2 = cache._cache_path(model="model1", messages=messages, temperature=0.5)
        assert path1 == path2

    def test_different_model_different_path(self, temp_dir: Path):
        """Test that different models produce different paths."""
        cache = PromptCache(temp_dir / "cache")
        messages = [{"role": "user", "content": "test"}]

        path1 = cache._cache_path(model="model1", messages=messages, temperature=0.5)
        path2 = cache._cache_path(model="model2", messages=messages, temperature=0.5)
        assert path1 != path2

    def test_different_messages_different_path(self, temp_dir: Path):
        """Test that different messages produce different paths."""
        cache = PromptCache(temp_dir / "cache")

        path1 = cache._cache_path(model="model", messages=[{"role": "user", "content": "test1"}], temperature=0.5)
        path2 = cache._cache_path(model="model", messages=[{"role": "user", "content": "test2"}], temperature=0.5)
        assert path1 != path2

    def test_different_temperature_different_path(self, temp_dir: Path):
        """Test that different temperatures produce different paths."""
        cache = PromptCache(temp_dir / "cache")
        messages = [{"role": "user", "content": "test"}]

        path1 = cache._cache_path(model="model", messages=messages, temperature=0.5)
        path2 = cache._cache_path(model="model", messages=messages, temperature=0.7)
        assert path1 != path2

    def test_none_temperature_different_path(self, temp_dir: Path):
        """Test that None temperature produces different path."""
        cache = PromptCache(temp_dir / "cache")
        messages = [{"role": "user", "content": "test"}]

        path1 = cache._cache_path(model="model", messages=messages, temperature=0.5)
        path2 = cache._cache_path(model="model", messages=messages, temperature=None)
        assert path1 != path2

    def test_path_is_file(self, temp_dir: Path):
        """Test that generated path is a file path."""
        cache = PromptCache(temp_dir / "cache")
        path = cache._cache_path(model="model", messages=[], temperature=None)
        assert path.suffix == ".json"


class TestPromptCacheGet:
    """Tests for get method."""

    def test_missing_cache(self, temp_dir: Path):
        """Test getting from missing cache returns None."""
        cache = PromptCache(temp_dir / "cache")
        result = cache.get(model="model", messages=[{"role": "user", "content": "test"}], temperature=0.5)
        assert result is None

    def test_get_cached_value(self, temp_dir: Path):
        """Test getting cached value."""
        cache = PromptCache(temp_dir / "cache")
        messages = [{"role": "user", "content": "test"}]

        # Set a value
        cache.set(model="model", messages=messages, temperature=0.5, content="cached response")

        # Get it back
        result = cache.get(model="model", messages=messages, temperature=0.5)
        assert result == "cached response"

    def test_corrupted_cache(self, temp_dir: Path):
        """Test getting from corrupted cache returns None."""
        cache = PromptCache(temp_dir / "cache")
        path = cache._cache_path(model="model", messages=[], temperature=0.5)
        path.write_text("not valid json")

        result = cache.get(model="model", messages=[], temperature=0.5)
        assert result is None

    def test_wrong_content_type(self, temp_dir: Path):
        """Test cache with wrong content type."""
        cache = PromptCache(temp_dir / "cache")
        path = cache._cache_path(model="model", messages=[], temperature=0.5)
        path.write_text(json.dumps({"wrong_key": "value"}))

        result = cache.get(model="model", messages=[], temperature=0.5)
        assert result is None


class TestPromptCacheSet:
    """Tests for set method."""

    def test_set_creates_file(self, temp_dir: Path):
        """Test that set creates a cache file."""
        cache = PromptCache(temp_dir / "cache")
        cache.set(model="model", messages=[{"role": "user", "content": "test"}], temperature=0.5, content="response")

        path = cache._cache_path(model="model", messages=[{"role": "user", "content": "test"}], temperature=0.5)
        assert path.exists()

    def test_set_stores_content(self, temp_dir: Path):
        """Test that set stores the content correctly."""
        cache = PromptCache(temp_dir / "cache")
        content = "This is a cached response"
        cache.set(model="model", messages=[], temperature=0.5, content=content)

        path = cache._cache_path(model="model", messages=[], temperature=0.5)
        data = json.loads(path.read_text())
        assert data["content"] == content

    def test_set_unicode_content(self, temp_dir: Path):
        """Test that set stores unicode content."""
        cache = PromptCache(temp_dir / "cache")
        content = "Unicode: 你好 🌍 émojis 🎉"
        cache.set(model="model", messages=[], temperature=0.5, content=content)

        result = cache.get(model="model", messages=[], temperature=0.5)
        assert result == content

    def test_set_multiline_content(self, temp_dir: Path):
        """Test that set stores multiline content."""
        cache = PromptCache(temp_dir / "cache")
        content = "Line 1\nLine 2\nLine 3"
        cache.set(model="model", messages=[], temperature=0.5, content=content)

        result = cache.get(model="model", messages=[], temperature=0.5)
        assert result == content

    def test_set_overwrites(self, temp_dir: Path):
        """Test that set overwrites existing cache."""
        cache = PromptCache(temp_dir / "cache")
        cache.set(model="model", messages=[], temperature=0.5, content="first")
        cache.set(model="model", messages=[], temperature=0.5, content="second")

        result = cache.get(model="model", messages=[], temperature=0.5)
        assert result == "second"
