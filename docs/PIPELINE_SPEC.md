# MusicVision Pipeline Specification

## Overview

MusicVision is a music video production pipeline that combines open-source AI tools to turn a song (audio + lyrics) into a complete music video. It wraps HuMo (ByteDance) for video generation and FLUX for reference image generation into an iterative, user-controlled workflow.

## System Requirements

- **Primary GPU**: NVIDIA RTX 5090 (32GB VRAM) — runs UNet/DiT for both FLUX and HuMo
- **Secondary GPU**: NVIDIA RTX 4080 (16GB VRAM) — offloads text encoders, VAE, Whisper, audio separator
- **Multi-GPU Strategy**: Proven in ComfyUI. DiT/UNet on GPU0 (5090), everything else on GPU1 (4080). This allows running HuMo-17B comfortably and FLUX-dev at full quality.
- **Model Swapping**: FLUX and HuMo run in different pipeline stages (not simultaneously). Within each stage, model components are split across both GPUs.
- **Storage**: ~50GB for model weights (FLUX + HuMo + Whisper + VAE + audio separator). Additional space for project assets.
- **Python**: 3.11+ (HuMo requirement)
- **CUDA**: 12.4+ with flash_attn 2.6.3

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        MusicVision                          │
├─────────────┬──────────────┬──────────────┬─────────────────┤
│  Stage 1    │  Stage 2     │  Stage 3     │  Stage 4        │
│  INTAKE &   │  IMAGE GEN & │  VIDEO GEN   │  ASSEMBLY &     │
│  SEGMENTATION│  STORYBOARD │  (HuMo)      │  EXPORT         │
├─────────────┼──────────────┼──────────────┼─────────────────┤
│ Whisper     │ FLUX         │ HuMo TIA     │ ffmpeg          │
│ ffmpeg      │ (LoRA)       │ (17B or 1.7B)│ XML/EDL gen     │
│ LLM (Claude)│              │              │                 │
└─────────────┴──────────────┴──────────────┴─────────────────┘
```

## Data Model

### Project Config (`project.yaml`)

```yaml
project:
  name: "My Music Video"
  created: "2026-02-23T12:00:00Z"
  
song:
  audio_file: "input/song.wav"
  lyrics_file: "input/lyrics.txt"       # User-provided or Whisper-generated
  bpm: 120                               # Auto-detected or user-specified
  duration_seconds: 180.0

style_sheet:
  visual_style: "cinematic, moody lighting, shallow depth of field"
  color_palette: "dark blues, warm amber highlights"
  aspect_ratio: "16:9"
  resolution: "1280x720"
  
  # Persistent elements injected into every image/video prompt
  characters:
    - id: "singer"
      description: "Young woman with short black hair, angular features"
      reference_image: "assets/characters/singer_ref.png"
      lora_path: "assets/loras/singer_v1.safetensors"  # Optional
      lora_weight: 0.8
      
  props:
    - id: "vintage_mic"
      description: "Chrome vintage ribbon microphone on a black stand"
      reference_image: "assets/props/mic_ref.png"
      
  settings:
    - id: "stage"
      description: "Dimly lit stage with red velvet curtains and haze"
      reference_image: "assets/settings/stage_ref.png"

humo:
  model_size: "17B"          # "17B" or "1.7B"
  resolution: "720p"         # "720p" or "480p"
  scale_a: 2.0               # Audio guidance strength
  scale_t: 7.5               # Text guidance strength
  denoising_steps: 50
  
flux:
  model: "flux-dev"          # or flux-schnell for faster iteration
  steps: 28
  guidance_scale: 3.5
```

### Scene Definition (`scenes.json`)

```json
{
  "scenes": [
    {
      "id": "scene_001",
      "order": 1,
      "time_start": 0.0,
      "time_end": 3.88,
      "type": "vocal",
      "lyrics": "Standing in the rain tonight",
      "audio_segment": "segments/scene_001.wav",
      
      "image_prompt": "A young woman with short black hair stands on a dimly lit stage with red velvet curtains, chrome vintage ribbon microphone, rain visible through a window behind her, moody lighting, shallow depth of field",
      "image_prompt_user_override": null,
      "reference_image": "images/scene_001.png",
      "image_approved": false,
      
      "video_prompt": "A close-up of a young woman with short black hair singing into a chrome vintage microphone on a dimly lit stage. She sways gently, rain streaks visible on a window behind her. The lighting shifts subtly with the music.",
      "video_prompt_user_override": null,
      "video_clip": null,
      "video_approved": false,
      
      "sub_clips": [],
      
      "characters": ["singer"],
      "props": ["vintage_mic"],
      "settings": ["stage"],
      
      "notes": ""
    }
  ]
}
```

### Key Design Decision: Sub-clips for Long Scenes

HuMo generates max ~3.88 seconds (97 frames @ 25fps). Scenes longer than 4s are automatically split:

```
Scene: 8 seconds (time 10.0 → 18.0)
  ├── sub_clip_a: frames 0-96   (10.0 → 13.88s)
  ├── sub_clip_b: frames 0-96   (13.88 → 17.76s)
  └── sub_clip_c: frames 0-30   (17.76 → 18.0s)  ← partial, padded or trimmed
```

Sub-clips share the same reference image but may get slightly varied video prompts (e.g., camera angle shift) to add visual interest within a scene.

## Stage 1: Intake & Segmentation

### Inputs
- Audio file (WAV/MP3/FLAC)
- Lyrics text file (optional — Whisper can transcribe)

### Process

1. **Audio Analysis**
   - Vocal separation using Kim_Vocal_2 (isolate vocals from instrumental)
   - BPM detection
   - Whisper transcription with word-level timestamps (if lyrics not provided)
   - User reviews/corrects lyrics and timestamps

2. **Scene Segmentation** (LLM-assisted)
   - Input: lyrics with timestamps, song structure (verse/chorus/bridge/instrumental)
   - Rules:
     - Minimum scene: 2 seconds
     - Maximum scene: 10 seconds
     - Prefer cuts on musical phrase boundaries
     - Instrumental sections get their own scenes (type: "instrumental")
     - Chorus repetitions can reuse imagery with variations
   - Output: `scenes.json` with timestamps and types
   - User reviews and adjusts scene boundaries

3. **Audio Slicing**
   - ffmpeg splits the full audio into per-scene segments
   - Each segment saved as WAV at original sample rate
   - Vocal-separated versions also sliced (for HuMo's audio input)

### Outputs
- `project.yaml` (initial)
- `scenes.json` (scene list with timestamps)
- `segments/` directory with audio clips
- `segments_vocal/` directory with vocal-only clips

## Stage 2: Image Generation & Storyboard

### Process

1. **Prompt Generation** (LLM-assisted)
   - For each scene, generate an image prompt that:
     - Describes the visual content matching the lyrics/mood
     - Injects style_sheet elements (characters, props, settings)
     - Maintains narrative coherence across scenes
     - Accounts for scene type (vocal scenes show singer; instrumental scenes show environment/abstract)

2. **Image Generation** (FLUX)
   - Generate reference image per scene using FLUX
   - Apply character LoRA if specified
   - Resolution: match HuMo input requirements (720p aspect ratio)
   - User can:
     - Edit the prompt and regenerate
     - Upload their own reference image
     - Regenerate with different seed
     - Mark as approved

3. **Storyboard View**
   - Display all scenes in order with:
     - Thumbnail of reference image
     - Timestamp range
     - Lyrics
     - Image prompt (editable)
     - Approval status
   - User approves entire storyboard before proceeding

### Outputs
- Updated `scenes.json` with image prompts and approval status
- `images/` directory with reference images

## Stage 3: Video Generation (HuMo)

### Process

1. **Video Prompt Generation** (LLM-assisted)
   - Generate detailed video conditioning text for each scene
   - More detailed than image prompt — describes motion, camera movement, expressions
   - Follows HuMo's preferred prompt style (see HUMO_REFERENCE.md)

2. **HuMo Inference** (TIA mode)
   - Input per scene: text prompt + reference image + audio segment
   - Config from `project.yaml` (resolution, guidance scales, steps)
   - For scenes > 4s: generate sub-clips sequentially

3. **Scene Review**
   - User reviews each generated video clip
   - Can: edit video prompt, change guidance scales, regenerate
   - Mark each scene as approved

### Outputs
- Updated `scenes.json` with video prompts and approval status
- `clips/` directory with video clips
- `clips/sub/` for sub-clips of long scenes

## Stage 4: Assembly & Export

### Process

1. **Concatenation**
   - Join all approved scene clips in order
   - For multi-sub-clip scenes, join sub-clips first
   - Handle transitions between clips (hard cut by default)

2. **Audio Sync**
   - Replace concatenated audio with original uncut song
   - Align scene boundaries to song timestamps
   - Trim/pad individual clips as needed for exact sync

3. **Export Formats**
   - **Rough Cut**: Single MP4 file, full song, all scenes assembled
   - **Individual Scenes**: Numbered clips with timecode in filename
     - Format: `scene_001_00m00s000_00m03s880.mp4`
   - **DaVinci Resolve Project**: XML (FCPXML) or EDL file with:
     - All clips on timeline at correct positions
     - Original audio on separate track
     - Scene markers/labels
     - Clips referenced by relative path for easy import

### Outputs
- `output/rough_cut.mp4`
- `output/scenes/` directory with individual clips
- `output/musicvision_project.fcpxml` (or `.edl`)
- `output/musicvision_project_notes.txt` (scene descriptions for editor reference)

## Directory Structure

```
my_music_video/
├── project.yaml
├── scenes.json
├── input/
│   ├── song.wav
│   └── lyrics.txt
├── assets/
│   ├── characters/
│   ├── props/
│   ├── settings/
│   └── loras/
├── segments/
│   ├── scene_001.wav
│   └── ...
├── segments_vocal/
│   ├── scene_001_vocal.wav
│   └── ...
├── images/
│   ├── scene_001.png
│   └── ...
├── clips/
│   ├── scene_001.mp4
│   ├── sub/
│   │   ├── scene_003_a.mp4
│   │   └── scene_003_b.mp4
│   └── ...
└── output/
    ├── rough_cut.mp4
    ├── scenes/
    │   └── scene_001_00m00s000_00m03s880.mp4
    ├── musicvision_project.fcpxml
    └── musicvision_project_notes.txt
```

## Module Structure (Python)

```
musicvision/
├── __init__.py
├── cli.py                    # CLI entry point (batch/headless)
├── app.py                    # Gradio UI entry point
├── config.py                 # Project/scene config loading
├── ui/
│   ├── __init__.py
│   ├── setup_tab.py          # Project setup, song upload, style sheet
│   ├── segmentation_tab.py   # Lyrics + scene boundary editing
│   ├── storyboard_tab.py     # Main grid: image gen, video gen, approve/reject
│   └── export_tab.py         # Assembly preview, export controls
├── intake/
│   ├── __init__.py
│   ├── audio_analysis.py     # BPM detection, vocal separation
│   ├── transcription.py      # Whisper transcription + alignment
│   └── segmentation.py       # LLM-assisted scene segmentation
├── imaging/
│   ├── __init__.py
│   ├── prompt_generator.py   # LLM-assisted image prompt generation
│   ├── flux_engine.py        # FLUX inference wrapper
│   └── storyboard.py         # Storyboard display/management
├── video/
│   ├── __init__.py
│   ├── prompt_generator.py   # LLM-assisted video prompt generation
│   ├── humo_engine.py        # HuMo inference wrapper
│   └── scene_manager.py      # Scene review/regeneration
├── assembly/
│   ├── __init__.py
│   ├── concatenator.py       # Clip joining + audio sync
│   ├── exporter.py           # FCPXML/EDL generation
│   └── timecode.py           # Timecode utilities
└── utils/
    ├── __init__.py
    ├── gpu.py                # Multi-GPU management: DiT→GPU0(5090), encoders/VAE→GPU1(4080)
    ├── audio.py              # ffmpeg wrappers for audio slicing
    └── paths.py              # Project directory management
```

## UI Architecture (Gradio)

### Tab 1: Project Setup
- Song upload (WAV/MP3/FLAC) + lyrics upload or paste
- Style sheet editor (visual style, color palette, characters, props, settings)
- LoRA management (upload, assign to characters)
- HuMo/FLUX config (model size, guidance scales, resolution)

### Tab 2: Segmentation
- Lyrics display with word-level timestamps (from Whisper)
- Editable scene boundary markers
- Scene list with type labels (vocal/instrumental)
- "Re-segment" button

### Tab 3: Storyboard (main workflow)
Grid layout — one row per scene:

| Column | Content | Interactive |
|--------|---------|------------|
| 1. Scene Info | Lyrics text (or "Instrumental"), timestamp range | Read-only |
| 2. Reference Image | Generated image thumbnail | Click to enlarge |
| 3. Image Prompt | Editable textbox with generated prompt | User editable + "Regenerate" button |
| 4. Video Conditioning | Editable textbox + LoRA selector dropdown | User editable + "Regenerate" button |
| 5. Video Clip | Generated video player | Click to play + "Regenerate" button |
| 6. Status | Approve/reject checkboxes for image and video | Toggle |

- Each row operates independently — regenerating scene 5 doesn't touch scene 4
- Batch operations: "Generate All Images", "Generate All Videos", "Approve All"
- Visual indicators: green checkmark (approved), yellow (pending), red (rejected/needs regen)

### Tab 4: Assembly & Export
- Full rough-cut video player with audio
- Scene-by-scene timeline scrubber
- Export controls:
  - Rough cut MP4
  - Individual scene clips (with timecoded filenames)
  - DaVinci Resolve project file (FCPXML/EDL)
  - Project notes export

## Open Questions / Future Work

- **Transitions**: Currently hard cuts only. Future: AI-generated transitions, crossfades.
- **Camera movement**: HuMo handles some implicit camera motion via prompting. Could add explicit camera control.
- **Batch rendering**: For cloud/multi-GPU setups, parallelize scene generation.
- **Style transfer**: Apply consistent color grading across all scenes pre-export.
- **Character consistency**: LoRA is the current approach. Could explore IP-Adapter or other identity preservation methods.
