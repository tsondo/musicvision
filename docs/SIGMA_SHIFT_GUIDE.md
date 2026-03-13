# HuMo Sigma Shift Guide

## How Sigma Shift Works

The sigma shift parameter warps the flow matching noise schedule:

```
σ_shifted = shift × σ / (1 + (shift - 1) × σ)
```

- **Higher shift** → more denoising steps allocated to high-noise (structural) region → more dynamic motion, but less fine-grained detail preservation
- **Lower shift** → more steps allocated to low-noise (detail) region → smoother motion, better face/lip detail, stronger reference image adherence

This directly interacts with HuMo's **time-adaptive CFG**, which prioritizes text guidance (semantic structure) in early steps and identity/audio preservation in later steps. Lower shift values give the model more "budget" in the detail-preservation phase where lip sync and facial fidelity are resolved.

### MusicVision Defaults

| Parameter | Current Default | Range in UI |
|-----------|----------------|-------------|
| `HumoConfig.shift` | 1.5 | 0.5 – 4.0 (step 0.5) |

The upstream Wan2.1 defaults are shift=5.0 (720p) or shift=3.0 (480p). HuMo's training used shift=8.0 for the UniPC scheduler. MusicVision's lower default of 1.5 reflects your empirical finding that performance/singing scenes benefit from a much lower shift.

---

## Sigma Shift vs Shot Type

The core tradeoff: close-ups need facial detail fidelity (low shift), while wide shots need coherent full-body/environment motion (higher shift).

| Shot Type | Recommended Shift | Rationale |
|-----------|:-----------------:|-----------|
| **ECU** (Extreme Close-Up: eyes, mouth) | 0.5 – 1.0 | Maximum detail budget. Facial micro-expressions and lip shapes are extremely sensitive to noise schedule. Any structural instability is immediately visible. |
| **CU** (Close-Up: head + shoulders) | 1.0 – 1.5 | Primary performance shot. Lip sync accuracy and identity preservation are critical. Your empirical sweet spot for stage singing. |
| **MCU** (Medium Close-Up: chest up) | 1.5 – 2.0 | Hand gestures and upper body motion become important. Slightly higher shift gives enough motion budget for arm/hand movement while keeping face readable. |
| **MS** (Medium Shot: waist up) | 2.0 – 2.5 | Body language and gesture dominate. Face is smaller in frame so slight detail loss is acceptable. Higher shift enables more natural body sway and arm movement. |
| **MWS** (Medium Wide Shot: knees up) | 2.5 – 3.0 | Full gesture range. Walking/swaying motion needs the dynamic range. Face is small enough that moderate detail loss doesn't hurt. |
| **WS** (Wide Shot: full body + environment) | 3.0 – 4.0 | Environment and full-body motion coherence matter most. Face is a small fraction of the frame. Higher shift prevents the "frozen statue" look where everything is too still. |

**Key insight**: For sub-clip continuity (last frame → next reference), lower shift values produce more stable handoff frames. If a long scene uses wide shots, consider slightly reducing shift from the table values to improve sub-clip transitions.

---

## Sigma Shift vs Camera Motion

Camera motion adds temporal complexity. The model must simultaneously generate subject motion AND camera motion. Higher shift gives more "motion budget" but can cause instability.

| Camera Motion | Recommended Shift | Notes |
|---------------|:-----------------:|-------|
| **Static** (locked tripod) | Use shot-type value | No additional motion budget needed. This is the most forgiving case — use the shot type recommendation directly. |
| **Slow push-in / pull-out** | Shot-type value | Gentle focal length change. Minimal additional complexity. Works well at any shift. |
| **Slow pan** (horizontal sweep) | +0.5 over shot-type | Pan requires coherent background generation across frames. Slight shift increase helps. |
| **Slow tilt** (vertical sweep) | +0.5 over shot-type | Similar to pan. Moderate increase. |
| **Tracking shot** (camera follows subject) | +0.5 to +1.0 over shot-type | Subject stays framed while background moves. Needs more dynamic motion. |
| **Dolly / steady-cam follow** | +1.0 over shot-type | Parallax and depth changes require significant motion coherence. |
| **Handheld / shake** | +0.5 over shot-type | Subtle instability. Don't over-shift — HuMo can generate natural micro-shake at moderate shift. Too high creates "drunk camera." |
| **Rapid zoom / whip pan** | 3.5 – 4.0 (regardless of shot type) | Extreme motion. Use sparingly — HuMo's 4s clips can't sustain this without artifacts. Consider splitting into a dedicated transition sub-clip. |

**Compound rule**: Start with the shot-type value, add the camera-motion offset, and cap at 4.0. Example: CU (1.0–1.5) + slow pan (+0.5) = 1.5–2.0.

**Important**: For performance scenes with lip sync, prefer camera motion that doesn't require a shift increase (static, slow push-in). If you need a tracking shot on a CU, accept that lip sync quality may slightly degrade — or use a wider framing.

---

## Sigma Shift vs Audio Type

HuMo's audio conditioning (Whisper embeddings → cross-attention) interacts with the noise schedule. Audio with clear, rhythmic vocal content benefits from lower shift because the model needs detail-phase steps to resolve the audio→lip mapping.

| Audio Type | Recommended Shift | `scale_a` Suggestion | Notes |
|------------|:-----------------:|:--------------------:|-------|
| **Clean singing** (standard vocal) | 1.0 – 1.5 | 4.0 – 5.5 | Your proven sweet spot. Clear pitch and rhythm give HuMo strong audio signal. Low shift preserves lip shape fidelity. |
| **Belting / powerful vocal** | 1.0 – 1.5 | 5.0 – 5.5 | High energy but still clean signal. Same range as standard singing. Higher `scale_a` can help match the intensity of mouth opening. |
| **Rap / fast delivery** | 1.5 – 2.0 | 5.0 – 5.5 | Rapid articulation needs slightly more motion budget to avoid "mushy mouth." The increased shift allows faster lip movement. |
| **Whispering / soft vocal** | 0.5 – 1.0 | 2.0 – 3.5 | Minimal mouth movement. Low shift preserves the subtle lip changes. Reduce `scale_a` — aggressive audio guidance on quiet audio creates artifacts. |
| **Spoken word / narration** | 1.0 – 1.5 | 3.0 – 4.5 | Similar to singing but usually less rhythmic. Moderate guidance. |
| **Screaming / harsh vocal** | 2.0 – 2.5 | 4.0 – 5.0 | Wide mouth, intense expression. Needs motion budget for the extreme facial deformation. Higher shift prevents the "frozen scream" look. |
| **Humming / closed-mouth** | 0.5 – 1.0 | 1.5 – 2.5 | Almost no lip movement. Extremely low shift + low `scale_a`. The scene's visual interest comes from body/environment, not lip sync. |
| **Instrumental** (no vocals) | 2.0 – 3.5 | 0.5 – 1.5 | No lip sync target. Higher shift for more dynamic visuals. Very low `scale_a` since there's nothing to sync to — but keep above 0 for rhythm-to-motion correlation. |
| **Mixed vocal + heavy instrumentation** | 1.5 – 2.0 | 4.0 – 5.0 | Noisy audio signal. Kim_Vocal_2 separation helps, but HuMo receives the full mix. Moderate shift hedges against audio noise. |

**Vocal separation note**: HuMo receives the full mix (not isolated vocals) because it was trained on mixed audio. For scenes with very heavy instrumentation competing with vocals, consider bumping `scale_a` up slightly to compensate for the weaker vocal signal in the mix.

---

## Combined Lookup: Quick Reference

For the most common MusicVision scenarios:

| Scenario | Shot | Camera | Audio | Shift | `scale_a` |
|----------|------|--------|-------|:-----:|:---------:|
| Standard performance verse | CU | Static | Singing | **1.0–1.5** | 5.0 |
| Chorus performance (energy) | MCU | Slow push | Belting | **1.5–2.0** | 5.5 |
| Rap verse | MCU | Static | Rap | **1.5–2.0** | 5.5 |
| Ballad intimate moment | ECU | Static | Whisper | **0.5–1.0** | 2.5 |
| Full stage performance | WS | Slow pan | Singing | **3.0–3.5** | 4.0 |
| Narrative b-roll | MS–WS | Tracking | Instrumental | **2.5–3.5** | 1.0 |
| Dramatic transition | WS | Whip/zoom | Instrumental | **3.5–4.0** | 0.5 |
| Emotional bridge | CU | Static | Soft vocal | **1.0** | 3.0 |
| Group performance | MWS | Slow pan | Singing | **2.5–3.0** | 4.5 |
| Screaming breakdown | MCU | Handheld | Scream | **2.5** | 5.0 |

---

## Implementation Notes

### Auto-selection in Pipeline

The planner values above can be encoded as a function for automatic sigma shift selection when the user hasn't set a per-scene override. Inputs: shot type (from image/video prompt keywords), scene type (vocal/instrumental), and camera motion hints.

```python
def recommend_sigma_shift(
    shot_type: str,       # "ecu", "cu", "mcu", "ms", "mws", "ws"
    camera_motion: str,   # "static", "slow_push", "slow_pan", "tracking", "handheld", "whip"
    audio_type: str,      # "singing", "belting", "rap", "whisper", "scream", "humming", "instrumental"
) -> float:
    """Return recommended sigma shift. See SIGMA_SHIFT_GUIDE.md for rationale."""
    ...
```

### Testing Protocol

When validating these recommendations:

1. **Seed-lock** — use the same seed across shift values so the only variable is the noise schedule
2. **Focus on lip sync first** — use SyncNet confidence score or manual frame-by-frame check on the mouth region
3. **Face detail second** — check identity similarity between reference image and output (ArcFace cosine similarity)
4. **Motion quality third** — subjective assessment of natural movement vs frozen/jittery artifacts
5. **Log latent statistics** — min/max/mean across denoising steps per your CLAUDE.md playbook

### Interaction with Other Parameters

- **`scale_a` (audio guidance)**: The shift and scale_a interact. Very low shift + very high scale_a can over-constrain the model, causing artifacts. The table above pairs them intentionally.
- **`scale_t` (text guidance)**: Keep at 5.0 for most cases. Only increase for scenes where prompt adherence is weak (complex scene descriptions).
- **Denoising steps**: Lower shift with fewer steps can under-denoise. If using shift < 1.5, prefer 40–50 steps rather than 30.
- **Sampler**: UniPC (order 2) is more tolerant of low shift than Euler because it uses higher-order correction. Stick with UniPC for shift < 2.0.
- **Resolution**: At 480p, face details are already limited — low shift has diminishing returns vs 720p. Consider allowing slightly higher shift at 480p preview resolution.
