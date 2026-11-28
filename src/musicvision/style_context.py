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


def _has_described_character(ss: StyleSheet) -> bool:
    return any(a.description for a in assets_of_type(ss, AssetType.CHARACTER))


def vocalist_voice(config: ProjectConfig) -> str | None:
    """'male' / 'female' when known, else None."""
    v = config.song.vocalist
    return v.voice if v is not None and v.voice in ("male", "female") else None


def vocalist_context_line(config: ProjectConfig) -> str | None:
    """Tell the LLM the singer's gender when no character asset describes the performer.

    A described character asset always wins (e.g. a deliberately cast performer
    lip-syncing someone else's voice)."""
    voice = vocalist_voice(config)
    if voice is None or _has_described_character(config.style_sheet):
        return None
    person = "man" if voice == "male" else "woman"
    return (
        f"Lead vocalist: {voice} voice. Any singer or performer shown is a {person}; "
        f"say so explicitly in the prompt."
    )


def template_subject(config: ProjectConfig, fallback: str) -> str:
    """Subject phrase for the no-LLM templates: first described character, else a
    performer of the vocalist's gender, else *fallback*."""
    for a in assets_of_type(config.style_sheet, AssetType.CHARACTER):
        if a.description:
            return a.description
    voice = vocalist_voice(config)
    if voice is not None:
        return fallback.replace("a performer", f"a {voice} singer", 1)
    return fallback


def template_setting(config: ProjectConfig, fallback: str) -> str:
    for a in assets_of_type(config.style_sheet, AssetType.LOCATION):
        if a.description:
            return a.description
    return fallback
