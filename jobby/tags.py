"""Tag normalization (jobby-plan.md item 3).

Database tags are free text ("JS", "JavaScript", "js"). Comparisons use the
canonical names from config/tag_aliases.yaml; tags listed under ``ignore`` are
never treated as a technology match on their own.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


class TagConfigError(RuntimeError):
    """Raised when the tag alias file is missing or invalid."""


@dataclass(frozen=True)
class Tags:
    aliases: dict[str, str]
    ignore: frozenset[str]

    def canonical(self, tag: str) -> str | None:
        key = str(tag).strip().lower()
        if not key:
            return None
        key = self.aliases.get(key, key)
        return None if key in self.ignore else key

    def canonical_set(self, tags: list[str]) -> set[str]:
        return {item for item in (self.canonical(tag) for tag in tags) if item}


def load_tags(path: str | Path) -> Tags:
    tag_path = Path(path)
    if not tag_path.is_file():
        raise TagConfigError(f"tag alias file not found: {tag_path}")
    raw = yaml.safe_load(tag_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TagConfigError(f"{tag_path}: top level must be a mapping")
    aliases = {
        str(key).strip().lower(): str(value).strip().lower()
        for key, value in (raw.get("aliases") or {}).items()
    }
    ignore = frozenset(str(item).strip().lower() for item in (raw.get("ignore") or []))
    return Tags(aliases=aliases, ignore=ignore)
