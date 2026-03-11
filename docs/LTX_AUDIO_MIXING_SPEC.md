# LTX-2 Generated Audio Mixing Spec

**Affected files:**
- `src/musicvision/video/ltx_video_engine.py` — decode and save generated audio
- `src/musicvision/video/base.py` — add `generated_audio_path` field to `VideoResult`
- `src/musicvision/models.py` — new `SceneAudioMode` enum, new fields on `Scene`
- `src/musicvision/assembly/concatenator.py` — call `build_mixed_audio()` before mux
- `src/musicvision/utils/audio.py` — new `build_mixed_audio()` function
- `src/musicvision/api/app.py` — PATCH endpoint accepts audio mix fields
- `frontend/src/api/types.ts` — new types
- `frontend/src/components/SceneRow.tsx` — audio mode selector + volume/fade controls

## Goal

LTX-Video 2 generates audio alongside video. Currently the audio latents are discarded. This spec adds the ability to:
1. Save LTX-2's generated audio as a WAV file alongside the video clip
2. Let the user choose per-scene whether to use the generated audio, the original song, or a mix
3. Control volume levels and fade in/out durations for the generated audio
4. During assembly, blend the generated audio with the (optionally ducked) original song

Use cases: LTX-2 generates dialogue, narration, sound effects, or atmospheric audio that should play over or instead of the music for specific scenes.

---

## LTX-2 Engine Changes

### Save generated audio

The pipeline currently runs with `output_type="latent"` and returns `(video_latents, audio_latents)`. Only video latents are decoded. Change to also decode audio latents and save as WAV.

In `ltx_video_engine.py`, after video decode and save:

```python
# --- Decode and save generated audio ---
generated_audio_path = None
audio_latents = result[1] if isinstance(result, (tuple, list)) and len(result) > 1 else None

if audio_latents is not None and hasattr(self._pipe, "audio_vae"):
    try:
        generated_audio_path = self._decode_and_save_audio(
            audio_latents, input.output_path
        )
    except Exception:
        log.warning("Audio decode failed — clip saved without generated audio", exc_info=True)
```

New method on `LtxVideoEngine`:

```python
def _decode_and_save_audio(
    self, audio_latents: "torch.Tensor", video_path: Path
) -> Path:
    """Decode audio latents via audio_vae and save as WAV alongside the video."""
    import torch
    import torchaudio

    audio_vae = self._pipe.audio_vae
    vae_device = self.device_map.encoder_device

    audio_vae.to(vae_device)
    audio_latents = audio_latents.to(device=vae_device, dtype=audio_vae.dtype)

    with torch.no_grad():
        # The exact decode API depends on the LTX-2 diffusers version.
        if hasattr(audio_vae, "decode"):
            waveform = audio_vae.decode(audio_latents).sample
        elif hasattr(self._pipe, "_decode_audio"):
            waveform = self._pipe._decode_audio(audio_latents)
        else:
            log.warning("No audio decode method found on audio_vae or pipeline")
            return None

    # waveform shape: (batch, channels, samples) or (channels, samples)
    if waveform.dim() == 3:
        waveform = waveform[0]

    waveform = waveform.clamp(-1, 1).cpu()

    # Save as WAV — .gen_audio.wav suffix
    audio_path = video_path.with_suffix(".gen_audio.wav")
    sample_rate = 16000  # LTX-2 audio VAE native rate

    torchaudio.save(str(audio_path), waveform, sample_rate)
    log.info("Generated audio saved: %s (%.2fs)", audio_path.name, waveform.shape[-1] / sample_rate)

    return audio_path
```

### File naming convention

```
clips/scene_003.mp4              ← video (silent)
clips/scene_003.gen_audio.wav    ← LTX-2 generated audio

clips/sub/scene_005_a.mp4
clips/sub/scene_005_a.gen_audio.wav
clips/sub/scene_005_b.mp4
clips/sub/scene_005_b.gen_audio.wav
```

### VideoResult update

Add to `base.py`:

```python
@dataclass
class VideoResult:
    video_path: Path
    frames_generated: int
    duration_seconds: float
    metadata: dict = field(default_factory=dict)
    generated_audio_path: Path | None = None  # NEW: LTX-2 generated audio
```

HVA and HuMo leave this as `None`. Only LTX-2 populates it.

When the API processes a video generation result, it persists the path:

```python
if result.generated_audio_path:
    scene.generated_audio = str(result.generated_audio_path.relative_to(paths.root))
```

---

## Data Model

### New enum

```python
class SceneAudioMode(str, Enum):
    SONG_ONLY = "song_only"           # Default: original song, no generated audio
    GENERATED_ONLY = "generated_only" # Only LTX-2 audio (song silent for this scene)
    MIX = "mix"                       # Generated audio layered over ducked song
```

### New fields on Scene

```python
class Scene(BaseModel):
    # ... existing fields ...

    # LTX-2 generated audio
    generated_audio: Optional[str] = None       # path to .gen_audio.wav

    # Audio mixing controls (only meaningful when generated_audio is not None)
    audio_mode: SceneAudioMode = SceneAudioMode.SONG_ONLY
    generated_audio_volume: float = 0.8         # 0.0–1.0, gen audio loudness
    song_duck_volume: float = 0.3               # 0.0–1.0, song volume when ducking
    audio_fade_in: float = 0.15                 # seconds, fade in for generated audio
    audio_fade_out: float = 0.15                # seconds, fade out for generated audio
    song_duck_fade_in: float = 0.3              # seconds, how fast song ducks down
    song_duck_fade_out: float = 0.3             # seconds, how fast song returns to full
```

### Field semantics

**Volume controls:**

| Field | Range | Default | Description |
|-------|-------|---------|-------------|
| `generated_audio_volume` | 0.0–1.0 | 0.8 | Peak volume of the LTX-2 generated audio |
| `song_duck_volume` | 0.0–1.0 | 0.3 | Volume the song ducks to during this scene |

**Fade controls:**

| Field | Range | Default | Description |
|-------|-------|---------|-------------|
| `audio_fade_in` | 0.0–2.0s | 0.15 | Generated audio fades in from silence over this duration |
| `audio_fade_out` | 0.0–2.0s | 0.15 | Generated audio fades out to silence at scene end |
| `song_duck_fade_in` | 0.0–2.0s | 0.3 | Song volume ramps down from full to `song_duck_volume` |
| `song_duck_fade_out` | 0.0–2.0s | 0.3 | Song volume ramps back up from `song_duck_volume` to full |

**Timing diagram for a MIX scene:**

```
Time:     |-- scene start -------------------- scene end --|
          |                                                 |
Song:     ████████▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓████████
          full    ↘ duck_fade_in  ducked        duck_fade_out ↗ full
          1.0      \___________ 0.3 ___________/             1.0

Gen audio:        ░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░
          0.0    fade_in ↗    0.8 (peak)    ↘ fade_out   0.0
```

The song duck begins slightly before the generated audio fade-in starts (by the difference in their fade durations), so the song is already quiet when the generated audio becomes audible. This is handled automatically by the ffmpeg filter timing.

**For GENERATED_ONLY mode:**

Same as MIX but `song_duck_volume = 0.0`. The song is fully silenced for this scene, with the duck fades providing a smooth transition in and out of silence.

**For SONG_ONLY mode (default):**

All mixing fields are ignored. The original song plays at full volume. No processing needed.

### Behavior constraints

- `audio_mode` is only meaningful when `generated_audio` is not None. For scenes rendered by HVA or HuMo (which produce no audio), the mode is ignored regardless of its value.
- Fades are clamped to half the scene duration. A 2-second scene cannot have a 1.5s fade in + 1.5s fade out. The assembly code clamps: `fade_in = min(fade_in, scene_duration * 0.4)` and similarly for fade_out. This prevents the fades from overlapping.
- Setting `audio_fade_in = 0` and `audio_fade_out = 0` gives a hard cut (no fade). This is available but not recommended — even 0.05s prevents audible clicks.

---

## Assembly Changes

### Current flow

```
scene clips (silent) → concat video → mux original song → rough_cut.mp4
```

### New flow

```
scene clips (silent) → concat video → build_mixed_audio() → mux mixed audio → rough_cut.mp4
```

If no scenes use generated audio (`all audio_mode == SONG_ONLY`), the existing fast path is taken — the original song is muxed directly with no processing. This is the common case and avoids unnecessary re-encoding of audio.

### `build_mixed_audio()` — new function in `utils/audio.py`

```python
def build_mixed_audio(
    original_audio: Path,
    scenes: list,           # Scene objects
    project_root: Path,
    output_path: Path,
) -> Path:
    """
    Build the final audio track by mixing LTX-2 generated audio
    over the original song with per-scene volume and fade control.

    Returns path to the mixed audio WAV (or the original audio path
    if no scenes need mixing).
    """
```

**Implementation strategy: ffmpeg filter_complex**

The entire mix is done in a single ffmpeg invocation using `amix`, `volume`, `afade`, and `adelay` filters. This avoids loading audio into Python (no numpy/scipy audio processing needed).

The filter graph construction:

```python
def _build_audio_filter(
    scenes: list,
    project_root: Path,
    song_duration: float,
) -> tuple[list[str], str]:
    """
    Build ffmpeg input args and filter_complex string for audio mixing.

    Returns:
        (input_args, filter_string)
        input_args: ["-i", "song.wav", "-i", "gen1.wav", "-i", "gen2.wav", ...]
        filter_string: the -filter_complex value
    """
    inputs = []       # ffmpeg -i arguments
    filters = []      # filter chain segments
    mix_inputs = []   # labels to feed into final amix

    # Input 0 is always the original song
    # We need to apply volume automation to the song: duck it during
    # scenes that have generated audio.

    # Build song volume automation using the volume filter with
    # enable expressions for each ducked region.
    # ffmpeg's volume filter supports time-based enable:
    #   volume=0.3:enable='between(t,10.5,15.2)'

    duck_regions = []  # (start, end, duck_volume, fade_in, fade_out)

    gen_audio_idx = 1  # ffmpeg input index (0 = song)

    for scene in scenes:
        if scene.audio_mode == "song_only":
            continue
        if not scene.generated_audio:
            continue

        gen_path = project_root / scene.generated_audio
        if not gen_path.exists():
            log.warning("Generated audio missing for %s: %s — skipping", scene.id, gen_path)
            continue

        s_start = scene.time_start
        s_end = scene.time_end
        s_dur = s_end - s_start

        # Clamp fades to prevent overlap
        max_fade = s_dur * 0.4
        fade_in = min(scene.audio_fade_in, max_fade)
        fade_out = min(scene.audio_fade_out, max_fade)
        duck_fade_in = min(scene.song_duck_fade_in, max_fade)
        duck_fade_out = min(scene.song_duck_fade_out, max_fade)

        duck_vol = 0.0 if scene.audio_mode == "generated_only" else scene.song_duck_volume

        # Song ducking region (starts duck_fade_in before scene, ends duck_fade_out after)
        duck_start = max(0, s_start - duck_fade_in)
        duck_end = min(song_duration, s_end + duck_fade_out)
        duck_regions.append((duck_start, duck_end, duck_vol, duck_fade_in, duck_fade_out))

        # Generated audio: delay to scene start, apply volume + fade
        inputs.extend(["-i", str(gen_path)])

        gen_label = f"[gen{gen_audio_idx}]"
        out_label = f"[g{gen_audio_idx}]"

        # Chain: volume → fade in → fade out → delay to scene position
        # adelay value is in milliseconds
        delay_ms = int(s_start * 1000)

        filter_parts = []

        # Volume
        vol = scene.generated_audio_volume
        filter_parts.append(f"volume={vol:.2f}")

        # Fade in (from start of the generated clip)
        if fade_in > 0:
            filter_parts.append(f"afade=t=in:d={fade_in:.3f}")

        # Fade out (at end of generated clip)
        if fade_out > 0:
            # afade out starts at (clip_duration - fade_out)
            filter_parts.append(f"afade=t=out:st={s_dur - fade_out:.3f}:d={fade_out:.3f}")

        # Delay to position in timeline + pad to song length
        filter_parts.append(f"adelay={delay_ms}|{delay_ms}")
        filter_parts.append(f"apad=whole_dur={song_duration}")

        chain = f"[{gen_audio_idx}]" + ",".join(filter_parts) + out_label
        filters.append(chain)
        mix_inputs.append(out_label)

        gen_audio_idx += 1

    if not mix_inputs:
        return None, None  # No mixing needed

    # Song volume automation
    # Use a chain of volume filters with enable expressions for each duck region.
    # Between duck regions the song plays at full volume.
    song_filters = []
    for i, (ds, de, dv, dfi, dfo) in enumerate(duck_regions):
        # For smooth ducking, we use two volume filters per region:
        # 1. Ramp down: from full to duck_volume over duck_fade_in
        # 2. Hold: duck_volume for the scene duration
        # 3. Ramp up: from duck_volume to full over duck_fade_out
        #
        # ffmpeg volume filter with enable + eval=frame gives per-sample control:
        #   volume='if(between(t,DS,DE), DV + (1-DV)*..., 1)':eval=frame
        #
        # Simpler approach: use a single volume expression that handles the ramp.
        # For each duck region, the volume at time t is:
        #   - t < duck_start: 1.0
        #   - duck_start <= t < duck_start + fade_in: lerp(1.0, duck_vol)
        #   - duck_start + fade_in <= t < duck_end - fade_out: duck_vol
        #   - duck_end - fade_out <= t < duck_end: lerp(duck_vol, 1.0)
        #   - t >= duck_end: 1.0

        # This is complex in a single volume expression. Simpler: use
        # afade filters on the song itself.
        pass

    # Practical approach: build the song ducking as a volume expression.
    # For N duck regions, the expression is a nested if() chain.
    # Each region: if(between(t, start, end), <ramp_expr>, <next>)
    #
    # Ramp expression for region (ds, de, dv, dfi, dfo):
    #   if(lt(t, ds+dfi),
    #      1 - (1-dv) * (t-ds)/dfi,        ← ramp down
    #      if(lt(t, de-dfo),
    #         dv,                            ← hold
    #         dv + (1-dv) * (t-(de-dfo))/dfo ← ramp up
    #      )
    #   )

    if duck_regions:
        expr_parts = []
        for ds, de, dv, dfi, dfo in duck_regions:
            ramp_down_end = ds + dfi if dfi > 0 else ds
            ramp_up_start = de - dfo if dfo > 0 else de

            if dfi > 0 and dfo > 0:
                region_expr = (
                    f"if(between(t,{ds:.3f},{de:.3f}),"
                    f"if(lt(t,{ramp_down_end:.3f}),"
                    f"1-(1-{dv:.2f})*(t-{ds:.3f})/{dfi:.3f},"
                    f"if(lt(t,{ramp_up_start:.3f}),"
                    f"{dv:.2f},"
                    f"{dv:.2f}+(1-{dv:.2f})*(t-{ramp_up_start:.3f})/{dfo:.3f}"
                    f")),%%NEXT%%)"
                )
            elif dfi > 0:
                region_expr = (
                    f"if(between(t,{ds:.3f},{de:.3f}),"
                    f"if(lt(t,{ramp_down_end:.3f}),"
                    f"1-(1-{dv:.2f})*(t-{ds:.3f})/{dfi:.3f},"
                    f"{dv:.2f}),%%NEXT%%)"
                )
            elif dfo > 0:
                region_expr = (
                    f"if(between(t,{ds:.3f},{de:.3f}),"
                    f"if(lt(t,{ramp_up_start:.3f}),"
                    f"{dv:.2f},"
                    f"{dv:.2f}+(1-{dv:.2f})*(t-{ramp_up_start:.3f})/{dfo:.3f}"
                    f"),%%NEXT%%)"
                )
            else:
                region_expr = f"if(between(t,{ds:.3f},{de:.3f}),{dv:.2f},%%NEXT%%)"

            expr_parts.append(region_expr)

        # Chain: nest the expressions, innermost returns 1.0 (full volume)
        vol_expr = "1.0"
        for part in reversed(expr_parts):
            vol_expr = part.replace("%%NEXT%%", vol_expr)

        song_chain = f"[0]volume='{vol_expr}':eval=frame[song]"
        filters.insert(0, song_chain)
        mix_inputs.insert(0, "[song]")
    else:
        mix_inputs.insert(0, "[0]")

    # Final amix: combine song + all generated audio tracks
    n_inputs = len(mix_inputs)
    mix_labels = "".join(mix_inputs)

    # amix with duration=first (match song length), normalize=0 (no auto-normalize)
    filters.append(f"{mix_labels}amix=inputs={n_inputs}:duration=first:normalize=0")

    filter_string = ";".join(filters)
    return inputs, filter_string
```

### Updated `assemble_rough_cut()`

```python
def assemble_rough_cut(scenes, paths, original_audio, approved_only=False):
    # ... existing clip collection logic ...

    # Check if any scenes need audio mixing
    needs_mixing = any(
        s.audio_mode != SceneAudioMode.SONG_ONLY
        and s.generated_audio is not None
        for s in ordered
    )

    if needs_mixing:
        mixed_audio = paths.output_dir / "_mixed_audio.wav"
        build_mixed_audio(original_audio, ordered, paths.root, mixed_audio)
        final_audio = mixed_audio
    else:
        final_audio = original_audio  # fast path: no mixing needed

    # ... existing concat + mux logic, using final_audio instead of original_audio ...

    output = paths.output_dir / "rough_cut.mp4"
    mux_video_audio(silent_cut, final_audio, output)

    # Clean up
    if needs_mixing and mixed_audio.exists():
        mixed_audio.unlink(missing_ok=True)
```

---

## FCPXML / EDL Export

### FCPXML

FCPXML supports multiple audio lanes. The generated audio should appear as a separate audio clip on a second audio lane, overlapping the song. This gives the user full manual control in DaVinci Resolve:

```xml
<spine>
  <!-- Video + song audio on lane 0 (existing) -->
  <asset-clip ref="r1" offset="0/25s" duration="100/25s" ...>
    <audio-channel-source srcCh="1,2" role="dialogue"/>
  </asset-clip>

  <!-- Generated audio on lane 1 (new) -->
  <audio role="music.music-2" offset="250/25s" duration="75/25s"
         ref="r_gen_scene_003" srcCh="1,2">
    <adjust-volume>
      <param name="amount" value="0.8dB"/>
    </adjust-volume>
  </audio>
</spine>
```

The volume and fade parameters from the Scene model are baked into the FCPXML as volume keyframes, giving the user a starting point they can fine-tune in Resolve.

### EDL

EDL has no multi-track audio support. Include a comment noting which scenes have generated audio:

```
* GENERATED AUDIO: scene_003 (mix, vol=0.8, duck=0.3, fade_in=0.15, fade_out=0.15)
```

The user would need to manually import the `.gen_audio.wav` files in Resolve. This is acceptable since EDL is the simpler format.

---

## Frontend Changes

### SceneRow audio controls

For scenes where `generated_audio` is not null, show an audio mixing panel. This appears as a collapsible section within the scene row, only visible for LTX-2 scenes that have generated audio.

```tsx
{scene.generated_audio && (
  <div className="audio-mix-controls">
    {/* Audio mode selector */}
    <select
      value={scene.audio_mode}
      onChange={(e) => onUpdate(scene.id, { audio_mode: e.target.value })}
      disabled={disabled}
    >
      <option value="song_only">Song Only</option>
      <option value="mix">Mix (Song + Generated)</option>
      <option value="generated_only">Generated Audio Only</option>
    </select>

    {scene.audio_mode !== "song_only" && (
      <>
        {/* Generated audio volume */}
        <label className="audio-control">
          <span>Gen Volume</span>
          <input type="range" min={0} max={1} step={0.05}
            value={scene.generated_audio_volume}
            onChange={(e) => onUpdate(scene.id, {
              generated_audio_volume: parseFloat(e.target.value)
            })}
          />
          <span className="audio-value">{Math.round(scene.generated_audio_volume * 100)}%</span>
        </label>

        {/* Song duck volume (only for mix mode) */}
        {scene.audio_mode === "mix" && (
          <label className="audio-control">
            <span>Song Volume</span>
            <input type="range" min={0} max={1} step={0.05}
              value={scene.song_duck_volume}
              onChange={(e) => onUpdate(scene.id, {
                song_duck_volume: parseFloat(e.target.value)
              })}
            />
            <span className="audio-value">{Math.round(scene.song_duck_volume * 100)}%</span>
          </label>
        )}

        {/* Fade controls */}
        <div className="audio-fades">
          <label className="audio-control">
            <span>Fade In</span>
            <input type="number" min={0} max={2} step={0.05}
              value={scene.audio_fade_in}
              onChange={(e) => onUpdate(scene.id, {
                audio_fade_in: parseFloat(e.target.value)
              })}
            />
            <span className="audio-unit">s</span>
          </label>
          <label className="audio-control">
            <span>Fade Out</span>
            <input type="number" min={0} max={2} step={0.05}
              value={scene.audio_fade_out}
              onChange={(e) => onUpdate(scene.id, {
                audio_fade_out: parseFloat(e.target.value)
              })}
            />
            <span className="audio-unit">s</span>
          </label>
        </div>

        {/* Playback preview: play just the generated audio */}
        <button
          className="btn btn-sm"
          onClick={() => playGeneratedAudio(scene.id)}
          title="Preview generated audio"
        >
          ▶ Preview Audio
        </button>
      </>
    )}
  </div>
)}
```

### Audio preview

A small inline audio player lets the user listen to the generated audio for a scene without running full assembly. This is just an `<audio>` element pointed at the `.gen_audio.wav` file via the existing file serving endpoint. No mixing — just the raw generated audio, so the user can decide whether it's worth including.

---

## Preview Player Integration

The Preview Playback spec (separate document) uses the original song as the audio master. When audio mixing is implemented, the preview player should be updated to:

1. For `song_only` scenes: play the original song (current behavior)
2. For `mix` / `generated_only` scenes: ideally play the mixed audio

**Practical approach**: The preview player continues using the original song for simplicity. The mixed audio is only heard in the assembled rough cut. Add a note in the preview UI: "Audio mixing is applied in the assembled output." This avoids real-time audio mixing in the browser, which is possible (Web Audio API) but complex and not worth the effort for a QA tool.

**Future enhancement**: If browser-based mixing becomes important, the Web Audio API can handle it — create a GainNode for the song and for each generated audio track, automate gain over time. But this is out of scope for the initial implementation.

---

## Implementation Order

1. **LTX-2 engine: decode and save audio** — Modify `generate()` to decode `result[1]` via `audio_vae` and save as `.gen_audio.wav`. Add `generated_audio_path` to `VideoResult`. Test: generate an LTX-2 clip and verify the WAV file appears with correct duration.

2. **Data model** — Add `SceneAudioMode` enum, new Scene fields, UpdateSceneRequest fields. Verify backward compat (all new fields have defaults, existing projects load without errors).

3. **API persistence** — When saving video gen results, persist `generated_audio` path. PATCH endpoint accepts audio mix fields.

4. **`build_mixed_audio()`** — Implement the ffmpeg filter graph builder. Test with a mock project: 3 scenes, middle scene has `audio_mode=mix`. Verify the output WAV has ducked song + generated audio at the correct position with fades.

5. **Assembly integration** — Wire `build_mixed_audio()` into `assemble_rough_cut()`. Verify the fast path still works (no mixing → original song muxed directly).

6. **Frontend** — Audio mode selector, volume sliders, fade inputs, inline audio preview.

7. **FCPXML export** — Generated audio as separate audio lane with volume keyframes.

---

## What NOT to Do

- No real-time audio mixing in the browser preview player. Assembly produces the mixed audio.
- No audio normalization or loudness metering. The user sets volumes manually. LUFS normalization is a Resolve finishing concern.
- No generated audio for HVA or HuMo scenes. Only LTX-2 produces audio. If other engines gain audio output in the future, the same `generated_audio` field and mixing controls apply — no architecture change needed.
- No audio effects (reverb, EQ, compression). This is a mixing level control, not an audio workstation. Effects go in Resolve.
- No per-sub-clip audio mixing controls. The mixing is per-scene. Sub-clips within a scene all use the same audio mode. If a scene has 3 sub-clips, their generated audio WAVs are concatenated before mixing, or each is individually positioned (the ffmpeg filter handles this via the sub-clip timestamps).
