# Scene Transitions Spec

**Affected files:**
- `src/musicvision/models.py` — new `TransitionType` enum, new fields on `Scene`
- `src/musicvision/assembly/concatenator.py` — transition-aware assembly
- `src/musicvision/utils/audio.py` — new `concat_videos_with_transitions()` function
- `src/musicvision/assembly/exporter.py` — FCPXML/EDL transition output
- `src/musicvision/api/app.py` — `PATCH /api/scenes/{id}` accepts transition fields
- `frontend/src/api/types.ts` — new types
- `frontend/src/components/SceneRow.tsx` — transition selector UI

**Depends on:** LTX_AUDIO_MIXING_SPEC.md must be implemented first. This spec assumes the `SceneAudioMode`, `generated_audio`, `build_mixed_audio()`, and related Scene fields from that spec already exist.

## Goal

Allow users to select a transition effect between scenes in the storyboard. The transition is applied during assembly using ffmpeg's `xfade` video filter. A small, reliable set of transitions — not a sprawling effects library.

---

## Transition Types

```python
class TransitionType(str, Enum):
    CUT = "cut"                     # Hard cut (current behavior, default)
    CROSSFADE = "crossfade"         # Dissolve / blend between scenes
    DIP_TO_BLACK = "dip_to_black"   # Fade out to black, fade in from black
    DIP_TO_WHITE = "dip_to_white"   # Fade out to white, fade in from white
    WIPE_LEFT = "wipe_left"         # Horizontal wipe, left to right
    WIPE_RIGHT = "wipe_right"       # Horizontal wipe, right to left
    FADE_IN = "fade_in"             # Black → scene (only valid on first scene)
    FADE_OUT = "fade_out"           # Scene → black (only valid on last scene)
```

These map directly to ffmpeg `xfade` transition names:

| TransitionType | ffmpeg xfade `transition=` |
|---------------|--------------------------|
| `crossfade` | `fade` |
| `dip_to_black` | `fadeblack` |
| `dip_to_white` | `fadewhite` |
| `wipe_left` | `wipeleft` |
| `wipe_right` | `wiperight` |

`fade_in` and `fade_out` use the `fade` video filter (not xfade) since they apply to a single clip, not a transition between two clips.

---

## Data Model Changes

### Scene model (`models.py`)

Add two fields to `Scene`:

```python
class Scene(BaseModel):
    # ... existing fields ...

    # Transition INTO this scene (from previous scene)
    transition_in: TransitionType = TransitionType.CUT
    transition_in_duration: float = 0.5  # seconds, range 0.1 – 2.0

    # Note: transition_in on the FIRST scene is interpreted as fade_in if set
    # to anything other than CUT. transition_in_duration on scene_001 controls
    # the fade-in duration.
```

The transition is stored on the **receiving** scene (the scene being transitioned INTO). This means:
- Scene 1 `transition_in = fade_in` → the video fades in from black
- Scene 5 `transition_in = crossfade` → scenes 4→5 crossfade
- Last scene has no special treatment unless you add `fade_out` to the assembly config (project-level, not per-scene)

### Why `transition_in` not `transition_out`

Storing on the receiving scene means each scene fully defines how it enters. No need to cross-reference the previous scene. The storyboard UI shows "how does this scene begin?" which is a natural mental model.

### UpdateSceneRequest

Add to `UpdateSceneRequest`:

```python
class UpdateSceneRequest(BaseModel):
    # ... existing fields ...
    transition_in: Optional[TransitionType] = None
    transition_in_duration: Optional[float] = None
```

### TypeScript types

```typescript
export type TransitionType =
  | "cut"
  | "crossfade"
  | "dip_to_black"
  | "dip_to_white"
  | "wipe_left"
  | "wipe_right"
  | "fade_in"
  | "fade_out";

// Add to Scene interface:
transition_in: TransitionType;
transition_in_duration: number;

// Add to UpdateSceneRequest:
transition_in?: TransitionType;
transition_in_duration?: number;
```

---

## Assembly Changes

### The duration problem

Transitions consume time from BOTH clips. A 0.5s crossfade between scene A (4s) and scene B (3s) means the assembled output is 6.5s, not 7s. The overlap is 0.5s where both clips are visible.

**This affects total video duration and therefore audio sync.**

The audio track (whether the original song or a mixed audio track from `build_mixed_audio()`) is muxed over the assembled video. If transitions shorten the video, it drifts out of sync.

### Solution: Freeze-extend the last clip

Each scene's video clip stays its original length. The `xfade` filter overlaps them, shortening the total. To compensate:

1. Compute `total_overlap = sum(t.transition_in_duration for t in scenes if t.transition_in != CUT)`
2. Extend the last clip by `total_overlap` seconds using ffmpeg's `tpad` filter (freeze last frame)
3. The assembled video duration now matches the sum of all scene durations, which matches the audio duration

```python
if total_overlap > 0:
    extended = _extend_clip_freeze(last_clip_path, total_overlap, fps)
    clip_paths[-1] = extended
```

### Interaction with LTX-2 audio mixing

**This spec assumes the LTX Audio Mixing Spec is already implemented.** The complete assembly pipeline has two independent processing axes:

- **Video axis**: scene clips → transitions (xfade) → freeze-extend → assembled silent video
- **Audio axis**: original song → `build_mixed_audio()` (if LTX scenes need mixing) → final audio track

These axes are independent. Video transitions do not affect audio mixing, and audio mixing does not affect video transitions. They converge at the final mux step.

**Audio positioning uses scene timestamps from `scenes.json`, not video frame positions.** Transitions create visual overlap between scenes but do not change when scenes logically start and end. A 0.5s crossfade between scenes at 10.0s and 14.0s means both scenes' audio is positioned at their original timestamps. Since the crossfade is a visual blend (where both scenes are partially visible), the audio timing is correct — there is no perceptible mismatch.

**The freeze-extend on the last clip is critical.** Without it, the video would be `total_overlap` seconds shorter than the audio. The mux uses `-shortest`, which would truncate the audio. Freeze-extend ensures `video_duration == audio_duration` regardless of how many transitions are used.

### concat_videos_with_transitions()

New function in `src/musicvision/utils/audio.py`:

```python
def concat_videos_with_transitions(
    clips: list[Path],
    transitions: list[dict],  # [{type: str, duration: float}, ...] — len = len(clips) - 1
    output_path: Path,
    fps: int = 25,
) -> Path:
    """
    Concatenate clips with xfade transitions between them.

    Uses ffmpeg filter_complex with chained xfade filters.
    All clips must have the same resolution and framerate.

    Args:
        clips: Ordered list of clip paths
        transitions: Transition spec for each boundary (clips[i] → clips[i+1])
                     Each dict: {"type": "fade"|"fadeblack"|..., "duration": float}
                     Use {"type": "fade", "duration": 0.0} for hard cuts within a
                     transition-enabled assembly.
        output_path: Where to write the result
        fps: Framerate for offset calculations

    Returns:
        Path to the output file
    """
```

The ffmpeg filter graph for N clips with transitions:

```
# 2 clips, 1 crossfade:
ffmpeg -i clip1.mp4 -i clip2.mp4 \
  -filter_complex "[0][1]xfade=transition=fade:duration=0.5:offset=3.5" \
  -c:v libx264 -crf 18 output.mp4

# 3 clips, 2 transitions:
ffmpeg -i clip1.mp4 -i clip2.mp4 -i clip3.mp4 \
  -filter_complex \
    "[0][1]xfade=transition=fade:duration=0.5:offset=3.5[v01]; \
     [v01][2]xfade=transition=fadeblack:duration=0.3:offset=6.7" \
  -c:v libx264 -crf 18 output.mp4
```

The `offset` for each xfade = cumulative video duration up to that point MINUS cumulative transition overlap:

```python
def _build_xfade_filter(
    clip_durations: list[float],
    transitions: list[dict],
) -> str:
    """Build ffmpeg filter_complex string for chained xfade filters."""
    if len(transitions) != len(clip_durations) - 1:
        raise ValueError("Need exactly len(clips)-1 transitions")

    filters = []
    cumulative = clip_durations[0]

    for i, t in enumerate(transitions):
        t_type = t["type"]
        t_dur = t["duration"]
        offset = cumulative - t_dur

        if i == 0:
            left = "[0]"
        else:
            left = f"[v{i-1:02d}]"

        right = f"[{i+1}]"

        if i < len(transitions) - 1:
            out = f"[v{i:02d}]"
        else:
            out = ""  # last filter outputs directly

        filters.append(
            f"{left}{right}xfade=transition={t_type}:duration={t_dur}:offset={offset:.4f}{out}"
        )

        cumulative += clip_durations[i + 1] - t_dur

    return ";".join(filters)
```

**Important**: xfade requires re-encoding. The output uses `-c:v libx264 -crf 18 -preset medium` for quality. This is slower than the current stream-copy concat, but transitions inherently require re-encoding.

### Updated assemble_rough_cut()

This is the **definitive** assembly function incorporating both this spec and the LTX Audio Mixing Spec. It replaces the existing `assemble_rough_cut()` entirely.

```python
TRANSITION_FFMPEG_MAP = {
    TransitionType.CROSSFADE: "fade",
    TransitionType.DIP_TO_BLACK: "fadeblack",
    TransitionType.DIP_TO_WHITE: "fadewhite",
    TransitionType.WIPE_LEFT: "wipeleft",
    TransitionType.WIPE_RIGHT: "wiperight",
}


def assemble_rough_cut(scenes, paths, original_audio, approved_only=False):
    ordered = sorted(scenes.scenes, key=lambda s: s.order)

    clip_paths = []
    transition_specs = []  # one per boundary between consecutive included clips
    included_scenes = []   # scenes that made it past the filter

    prev_included = False
    for scene in ordered:
        if approved_only and scene.video_status != ApprovalStatus.APPROVED:
            continue

        clip = _resolve_scene_clip(scene, paths)
        if clip is None:
            log.warning("Scene %s has no clip — skipping", scene.id)
            continue

        clip_paths.append(clip)
        included_scenes.append(scene)

        # Build transition list (one entry per boundary between consecutive clips)
        if prev_included and scene.transition_in != TransitionType.CUT:
            transition_specs.append({
                "type": TRANSITION_FFMPEG_MAP[scene.transition_in],
                "duration": scene.transition_in_duration,
            })
        elif prev_included:
            transition_specs.append(None)  # hard cut

        prev_included = True

    if not clip_paths:
        raise RuntimeError(
            "No clips available to assemble. "
            "Run video generation first, or disable approved_only."
        )

    paths.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Assemble video (silent) ──────────────────────────
    has_transitions = any(t is not None for t in transition_specs)

    if has_transitions:
        total_overlap = sum(
            t["duration"] for t in transition_specs if t is not None
        )

        # Freeze-extend last clip to compensate for transition overlap.
        # This ensures total video duration == sum of scene durations == audio duration.
        if total_overlap > 0:
            clip_paths[-1] = _extend_clip_freeze(clip_paths[-1], total_overlap, fps=25)

        # For boundaries that are hard cuts within a transition-enabled assembly,
        # use zero-duration xfade (effectively a cut in the filter graph).
        effective_transitions = [
            t if t is not None else {"type": "fade", "duration": 0.0}
            for t in transition_specs
        ]

        silent_cut = paths.output_dir / "_rough_cut_silent.mp4"
        concat_videos_with_transitions(clip_paths, effective_transitions, silent_cut)

    else:
        # Fast path: concat demuxer (stream copy, no re-encode)
        if len(clip_paths) == 1:
            silent_cut = clip_paths[0]
        else:
            silent_cut = paths.output_dir / "_rough_cut_silent.mp4"
            concat_videos(clip_paths, silent_cut)

    # ── Step 2: Build audio track ────────────────────────────────
    # From LTX Audio Mixing Spec: check if any scenes need audio mixing.
    # If so, build a mixed audio track. Otherwise use the original song.
    needs_mixing = any(
        getattr(s, "audio_mode", "song_only") != "song_only"
        and getattr(s, "generated_audio", None) is not None
        for s in included_scenes
    )

    if needs_mixing:
        mixed_audio = paths.output_dir / "_mixed_audio.wav"
        build_mixed_audio(original_audio, included_scenes, paths.root, mixed_audio)
        final_audio = mixed_audio
    else:
        final_audio = original_audio

    # ── Step 3: Mux video + audio ────────────────────────────────
    output = paths.output_dir / "rough_cut.mp4"
    mux_video_audio(silent_cut, final_audio, output)

    # ── Step 4: Sync check ───────────────────────────────────────
    try:
        video_dur = get_duration(output)
        audio_dur = get_duration(final_audio)
        fps = 25
        tolerance = 1.0 / fps
        drift = abs(video_dur - audio_dur)

        if drift > tolerance:
            log.warning(
                "SYNC WARNING: video=%.4fs, audio=%.4fs, drift=%.4fs (%.1f frames)",
                video_dur, audio_dur, drift, drift * fps,
            )
    except Exception as exc:
        log.debug("Could not verify sync: %s", exc)

    # ── Step 5: Cleanup ──────────────────────────────────────────
    for tmp in [
        paths.output_dir / "_rough_cut_silent.mp4",
        paths.output_dir / "_mixed_audio.wav",
    ]:
        if tmp.exists() and tmp != clip_paths[0]:
            tmp.unlink(missing_ok=True)

    # Clean up freeze-extended clips
    for cp in clip_paths:
        if "_extended" in cp.name and cp.exists():
            cp.unlink(missing_ok=True)

    log.info("Rough cut saved: %s (%d clips)", output, len(clip_paths))
    return output
```

### _extend_clip_freeze()

New helper in `utils/audio.py`:

```python
def _extend_clip_freeze(clip_path: Path, extend_seconds: float, fps: int) -> Path:
    """
    Extend a video clip by freezing its last frame for the given duration.

    Uses ffmpeg tpad filter. Returns path to the extended clip (new file).
    The original clip is not modified.
    """
    extended_path = clip_path.with_name(clip_path.stem + "_extended.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(clip_path),
        "-vf", f"tpad=stop_mode=clone:stop_duration={extend_seconds:.4f}",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-an",
        str(extended_path),
    ]

    subprocess.run(cmd, capture_output=True, check=True)
    return extended_path
```

### FCPXML transition output

In `exporter.py`, FCPXML 1.9 supports transitions between clips:

```xml
<transition name="Cross Dissolve" offset="175/25s" duration="12/25s">
    <filter-video ref="r_dissolve" name="Cross Dissolve"/>
</transition>
```

Map TransitionType to FCPXML built-in transition names:

| TransitionType | FCPXML transition |
|---------------|-------------------|
| crossfade | Cross Dissolve |
| dip_to_black | Fade to Color (black) |
| dip_to_white | Fade to Color (white) |
| wipe_left | Wipe |
| wipe_right | Wipe (reverse) |

The FCPXML exporter should insert `<transition>` elements between `<asset-clip>` elements, adjusting the timeline offset to account for the overlap duration.

### EDL transition output

EDL supports dissolves natively with the `D` edit type (instead of `C` for cut):

```
002  AX       V     D  012   001:23:00 001:23:12 002:15:00 002:15:12
```

The `012` is the transition duration in frames. Only dissolve is standard in EDL; wipes require CMX-style extensions. For simplicity, map all non-cut transitions to dissolve in EDL and include a comment noting the intended transition type.

---

## Frontend Changes

### SceneRow transition selector

Add a compact transition selector between each pair of scenes. This appears in the storyboard as a small row or inline control between scene rows.

Best location: a narrow "transition bar" rendered between `SceneRow` components in `Storyboard.tsx`:

```tsx
{scenes.map((scene, i) => (
  <React.Fragment key={scene.id}>
    {i > 0 && (
      <TransitionBar
        sceneId={scene.id}
        transition={scene.transition_in}
        duration={scene.transition_in_duration}
        onChange={(type, dur) => onUpdate(scene.id, {
          transition_in: type,
          transition_in_duration: dur,
        })}
        disabled={pipelineRunning}
      />
    )}
    <SceneRow ... />
  </React.Fragment>
))}
```

### TransitionBar component

Minimal: a single row with a dropdown for transition type and a small number input for duration.

```tsx
function TransitionBar({ sceneId, transition, duration, onChange, disabled }) {
  return (
    <div className="transition-bar">
      <select
        value={transition}
        onChange={(e) => onChange(e.target.value, duration)}
        disabled={disabled}
      >
        <option value="cut">Hard Cut</option>
        <option value="crossfade">Crossfade</option>
        <option value="dip_to_black">Dip to Black</option>
        <option value="dip_to_white">Dip to White</option>
        <option value="wipe_left">Wipe Left</option>
        <option value="wipe_right">Wipe Right</option>
      </select>
      {transition !== "cut" && (
        <input
          type="number"
          min={0.1}
          max={2.0}
          step={0.1}
          value={duration}
          onChange={(e) => onChange(transition, parseFloat(e.target.value))}
          disabled={disabled}
          className="transition-duration"
        />
      )}
    </div>
  );
}
```

CSS: thin horizontal bar, muted background, centered between scene rows. Should look like a connector, not a full row.

---

## Implementation Order

**This spec is implemented AFTER the LTX Audio Mixing Spec.** The assembly code builds on `build_mixed_audio()` and the `SceneAudioMode` / `generated_audio` fields from that spec.

1. **Data model** — Add `TransitionType` enum, Scene fields (`transition_in`, `transition_in_duration`), UpdateSceneRequest fields. Run existing tests to confirm backward compat (new fields have defaults).

2. **`_extend_clip_freeze()`** — New helper in `utils/audio.py`. Test: extend a 3s clip by 1s, verify output is 4s with frozen last frame.

3. **`concat_videos_with_transitions()`** — New function in `utils/audio.py`. Unit test with 2-3 short test clips and known durations. Verify total duration = sum(clips) - sum(overlaps) + freeze_extension.

4. **`assemble_rough_cut()` replacement** — Replace the existing function with the definitive version from this spec. It handles all four cases:
   - No transitions, no audio mixing → fast path (stream-copy concat + original song mux)
   - No transitions, with audio mixing → stream-copy concat + `build_mixed_audio()` + mux
   - Transitions, no audio mixing → xfade concat + freeze-extend + original song mux
   - Transitions + audio mixing → xfade concat + freeze-extend + `build_mixed_audio()` + mux

5. **API update** — PATCH endpoint accepts `transition_in` and `transition_in_duration`. Verify roundtrip.

6. **Frontend** — TransitionBar component, type definitions, storyboard integration.

7. **FCPXML/EDL export** — Transition elements in timeline export. Test import in DaVinci Resolve.

---

## What NOT to Do

- No AI-generated transitions (morph between scenes). That's a future feature requiring a separate model.
- No per-sub-clip transitions. Transitions are between SCENES only. Sub-clip joins within a scene remain hard cuts (they're supposed to look continuous).
- No audio crossfades in the rough cut. Audio mixing (ducking, LTX-2 generated audio) is handled by `build_mixed_audio()` from the LTX Audio Mixing Spec. Scene transitions are video-only. Audio crossfades between scenes are a DaVinci Resolve finishing concern.
- No transition preview in the browser. The user selects a type, assembles, and checks the result. Live preview of ffmpeg filters in-browser is not feasible.
- Do not modify `build_mixed_audio()` or the LTX audio mixing logic. Audio uses scene timestamps from `scenes.json` and is independent of video transitions.
