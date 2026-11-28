"""Shared style-sheet → prompt context for the image and video prompt generators.

Reads the unified ``StyleSheet.assets`` list (characters, props, locations).
Legacy ``characters``/``props``/``settings`` entries are migrated into
``assets`` when the style sheet loads, so ``assets`` is the single source.
"""

from __future__ import annotations

from musicvision.models import AssetDef, AssetType, ProjectConfig, StyleSheet

_ASSET_LABELS = {
    AssetType.CHARACTER: "Character",
    AssetType.PROP: "Prop",
    AssetType.LOCATION: "Setting",
}


def assets_of_type(ss: StyleSheet, asset_type: AssetType) -> list[AssetDef]:
    return [a for a in ss.assets if a.asset_type == asset_type]


def asset_context_lines(ss: StyleSheet) -> list[str]:
    """One line per described asset, characters first, for the LLM style context."""
    lines: list[str] = []
    for asset_type, label in _ASSET_LABELS.items():
        for a in assets_of_type(ss, asset_type):
            if not a.description:
                continue
            name = f" ({a.name})" if a.name and a.name != a.id else ""
            lines.append(f"{label} — {a.id}{name}: {a.description}")
    return lines


def template_subject(config: ProjectConfig, fallback: str) -> str:
    """Subject phrase for the no-LLM templates: first described character, else *fallback*."""
    for a in assets_of_type(config.style_sheet, AssetType.CHARACTER):
        if a.description:
            return a.description
    return fallback


def template_setting(config: ProjectConfig, fallback: str) -> str:
    for a in assets_of_type(config.style_sheet, AssetType.LOCATION):
        if a.description:
            return a.description
    return fallback
