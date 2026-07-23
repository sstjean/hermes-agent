"""Tests for the config-layer memory-provider helpers (local patch #12 / upstream #5688).

These lock the setter/reader contract at the config seam:

- ``set_active_memory_providers`` writes ``memory.providers`` canonically and
  mirrors the legacy singular ``memory.provider`` only when exactly one
  provider is active — killing the masking bug by construction.
- ``get_active_memory_providers`` resolves the ordered list with a legacy
  single-string fallback (legacy compat).
"""

from hermes_cli.config import (
    get_active_memory_providers,
    set_active_memory_providers,
)


class TestGetActiveMemoryProviders:
    def test_reads_ordered_list(self):
        cfg = {"memory": {"providers": ["index", "holographic"]}}
        assert get_active_memory_providers(cfg) == ["index", "holographic"]

    def test_legacy_singular_fallback(self):
        cfg = {"memory": {"provider": "mem0"}}
        assert get_active_memory_providers(cfg) == ["mem0"]

    def test_list_takes_precedence_over_legacy_singular(self):
        cfg = {"memory": {"providers": ["index"], "provider": "mem0"}}
        assert get_active_memory_providers(cfg) == ["index"]

    def test_empty_and_blank_entries_dropped(self):
        cfg = {"memory": {"providers": ["", "  ", "index"]}}
        assert get_active_memory_providers(cfg) == ["index"]

    def test_missing_memory_section(self):
        assert get_active_memory_providers({}) == []

    def test_non_dict_config(self):
        assert get_active_memory_providers(None) == []  # type: ignore[arg-type]


class TestSetActiveMemoryProviders:
    def test_single_provider_mirrors_legacy_singular(self):
        cfg: dict = {}
        set_active_memory_providers(cfg, ["mem0"])
        assert cfg["memory"]["providers"] == ["mem0"]
        assert cfg["memory"]["provider"] == "mem0"

    def test_multiple_providers_clears_legacy_singular(self):
        cfg: dict = {}
        set_active_memory_providers(cfg, ["index", "holographic"])
        assert cfg["memory"]["providers"] == ["index", "holographic"]
        assert cfg["memory"]["provider"] == ""

    def test_empty_clears_both(self):
        cfg = {"memory": {"providers": ["index"], "provider": "index"}}
        set_active_memory_providers(cfg, [])
        assert cfg["memory"]["providers"] == []
        assert cfg["memory"]["provider"] == ""

    def test_masking_killed_by_construction(self):
        """Concrete Given: a pre-existing non-empty memory.providers list must
        NOT mask a newly-configured provider.

        Start from ``providers: [holographic]`` already in config, then a setup
        hook configures ``mem0`` via the canonical setter. The effective set
        must become ``[mem0]`` — NOT silently remain ``[holographic]``. Starts
        from a NON-EMPTY list so it exercises the masking path, not a vacuous
        pass.
        """
        cfg = {"memory": {"providers": ["holographic"]}}
        set_active_memory_providers(cfg, ["mem0"])
        assert get_active_memory_providers(cfg) == ["mem0"]
        assert cfg["memory"]["providers"] == ["mem0"]
        assert cfg["memory"]["provider"] == "mem0"

    def test_blank_entries_dropped_on_write(self):
        cfg: dict = {}
        set_active_memory_providers(cfg, ["  ", "index", ""])
        assert cfg["memory"]["providers"] == ["index"]
        assert cfg["memory"]["provider"] == "index"

    def test_preserves_other_memory_keys(self):
        cfg = {"memory": {"memory_enabled": True, "mem0": {"k": "v"}}}
        set_active_memory_providers(cfg, ["mem0"])
        assert cfg["memory"]["memory_enabled"] is True
        assert cfg["memory"]["mem0"] == {"k": "v"}
        assert cfg["memory"]["providers"] == ["mem0"]

    def test_roundtrip_get_after_set(self):
        cfg: dict = {}
        set_active_memory_providers(cfg, ["index", "holographic"])
        assert get_active_memory_providers(cfg) == ["index", "holographic"]
