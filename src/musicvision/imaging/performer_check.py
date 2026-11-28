"""Vision check: does a performance-scene image show a singer of the vocalist's gender?

Uses the configured LLM with an image attached (vLLM Qwen3.6-VL or Claude).
Best effort: if the LLM is unreachable or refuses images, the check reports
``unavailable`` and nothing is flagged.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from musicvision.imaging.prompt_generator import _resolve_treatment
from musicvision.models import ProjectConfig, Scene, SceneTreatment
from musicvision.style_context import vocalist_context_line, vocalist_voice

log = logging.getLogger(__name__)

NOTE_PREFIX = "[auto] Performer check:"

PERFORMER_CHECK_SYSTEM = (
    "You inspect one frame from a music video. Identify the main performer: the most "
    "prominent person singing or performing. Reply with JSON only, no prose: "
    '{"main_performer": "man" | "woman" | "none" | "unclear"}. '
    'Use "none" if no person is visible and "unclear" if you cannot tell.'
)

_SEEN_FOR_VOICE: dict[str, Literal["man", "woman"]] = {"male": "man", "female": "woman"}


class PerformerCheck(BaseModel):
    status: Literal["match", "mismatch", "no_person", "unclear", "unavailable"]
    expected: Literal["man", "woman"]
    seen: str | None = None


def expected_performer(scene: Scene, config: ProjectConfig) -> Literal["man", "woman"] | None:
    """The performer a scene's image should show, or None when the check doesn't apply.

    Applies to performance-treatment scenes when the vocalist's voice is known
    and no character asset describes the performer (a cast character wins)."""
    voice = vocalist_voice(config)
    if voice is None or vocalist_context_line(config) is None:
        return None
    if _resolve_treatment(scene, config) != SceneTreatment.PERFORMANCE.value:
        return None
    return _SEEN_FOR_VOICE[voice]


def check_performer(image_path: Path, expected: Literal["man", "woman"], client=None) -> PerformerCheck:
    """Ask the vision LLM who the main performer is and compare with *expected*."""
    from musicvision.llm import get_client

    try:
        reply = (client or get_client()).chat(
            PERFORMER_CHECK_SYSTEM, "Who is the main performer in this frame?", images=[image_path],
        )
    except Exception as exc:  # unreachable server, vision disabled, etc.
        log.warning("Performer check unavailable for %s: %s", image_path.name, exc)
        return PerformerCheck(status="unavailable", expected=expected)

    match = re.search(r"\{.*\}", reply, re.DOTALL)
    try:
        seen = json.loads(match.group(0))["main_performer"] if match else None
    except (ValueError, KeyError, TypeError):
        seen = None
    if seen not in ("man", "woman", "none", "unclear"):
        log.warning("Performer check: unparseable reply for %s: %r", image_path.name, reply[:200])
        return PerformerCheck(status="unclear", expected=expected, seen=None)
    if seen == "none":
        return PerformerCheck(status="no_person", expected=expected, seen=seen)
    if seen == "unclear":
        return PerformerCheck(status="unclear", expected=expected, seen=seen)
    return PerformerCheck(status="match" if seen == expected else "mismatch", expected=expected, seen=seen)


def apply_check_to_scene(scene: Scene, check: PerformerCheck | None) -> dict | None:
    """Update the scene's auto note from a check; return a job warning on mismatch.

    Replaces any previous auto note so re-generating doesn't pile them up."""
    kept = [line for line in scene.notes.splitlines() if not line.startswith(NOTE_PREFIX)]
    if check is not None and check.status == "mismatch":
        kept.append(
            f"{NOTE_PREFIX} image shows a {check.seen}, but the lead vocalist is a {check.expected}. Review."
        )
    scene.notes = "\n".join(kept)
    if check is None or check.status != "mismatch":
        return None
    return {
        "code": "performer_mismatch",
        "scene_id": scene.id,
        "message": f"{scene.id}: image shows a {check.seen}; the lead vocalist is a {check.expected}.",
        "detail": check.model_dump(),
    }


def run_performer_check(scene: Scene, config: ProjectConfig, image_path: Path) -> dict | None:
    """Check one freshly generated image and record the result on the scene."""
    expected = expected_performer(scene, config)
    if expected is None:
        return apply_check_to_scene(scene, None)  # clears a stale note, e.g. after casting a character
    return apply_check_to_scene(scene, check_performer(image_path, expected))
