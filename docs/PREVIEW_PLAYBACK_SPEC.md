# In-Browser Preview Playback Spec

**Affected files:**
- `frontend/src/components/PreviewPlayer.tsx` — new component
- `frontend/src/components/Storyboard.tsx` — integration
- `frontend/src/App.tsx` — state management
- `frontend/src/App.css` — player styles
- `src/musicvision/api/app.py` — new endpoint for serving audio with byte-range support

## Goal

Let the user play back scene clips sequentially in the browser, synced to the original song audio, to check lip sync quality and scene continuity without exporting to DaVinci Resolve. This is a QA tool, not a video editor.

---

## Design Principles

1. **Use the browser's native `<video>` and `<audio>` elements.** No custom video decoders, no WebGL, no wasm. Browser-native playback is the most reliable path.
2. **Scene clips already exist as MP4 files** served by the FastAPI static file middleware. No new encoding step needed.
3. **Audio sync is achieved by playing the original song audio track and swapping video clips at scene boundaries.** The audio element is the master clock; video elements follow it.
4. **Lip sync quality assessment requires frame-accurate sync.** The `timeupdate` event on `<audio>` fires ~4x/sec (250ms) which is too coarse. Use `requestAnimationFrame` polling on `audio.currentTime` for ~16ms precision.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│ PreviewPlayer                                │
│                                              │
│  ┌─────────────────────────────────────┐     │
│  │ <video> element (current scene clip) │     │
│  │ Muted. Playback rate locked to 1x.  │     │
│  └─────────────────────────────────────┘     │
│                                              │
│  ┌─────────────────────────────────────┐     │
│  │ <audio> element (original song)      │     │
│  │ Master clock. Drives scene switching.│     │
│  └─────────────────────────────────────┘     │
│                                              │
│  Transport: [◀◀] [▶/⏸] [▶▶] [seek bar]     │
│  Info: Scene 5/12 | 1:23.4 / 3:45.0         │
│  Controls: [Play from scene N] [Loop scene]  │
└─────────────────────────────────────────────┘
```

### Why separate audio + video elements

The scene clips contain generation audio (the scene's audio segment), not the original song. Using the clip's built-in audio would produce audible splice artifacts at scene boundaries. Instead:

- The `<video>` element is **muted**. It shows the video only.
- The `<audio>` element plays the **original uncut song**. It's the timing master.
- When audio playback crosses a scene boundary, the video element switches to the next scene's clip.

This mirrors what the final rough cut does (original audio over concatenated video) but without requiring an actual assembly step.

---

## Scene Clip Resolution

For each scene, resolve the best available clip in priority order:

```typescript
function resolveClipUrl(scene: Scene): string | null {
  // Prefer upscaled > raw > sub-clip joined
  if (scene.upscaled_clip) return fileUrl(scene.upscaled_clip);
  if (scene.video_clip) return fileUrl(scene.video_clip);
  return null;
}
```

Scenes without a clip are skipped during playback (audio continues, video shows a "no clip" placeholder or the reference image).

---

## Sync Strategy

### Master clock: audio.currentTime

The `<audio>` element's `currentTime` is polled every animation frame. When it crosses a scene boundary, the video element is updated.

```typescript
const rafRef = useRef<number>(0);
const audioRef = useRef<HTMLAudioElement>(null);
const videoRef = useRef<HTMLVideoElement>(null);

// Scene boundary lookup (sorted by time_start)
const sceneBoundaries = useMemo(() =>
  scenes
    .filter(s => resolveClipUrl(s) !== null)
    .map(s => ({ id: s.id, start: s.time_start, end: s.time_end, clipUrl: resolveClipUrl(s)! }))
    .sort((a, b) => a.start - b.start),
  [scenes]
);

// Find which scene should be active at a given time
function findActiveScene(t: number): number {
  for (let i = sceneBoundaries.length - 1; i >= 0; i--) {
    if (t >= sceneBoundaries[i].start) return i;
  }
  return 0;
}

// Animation frame sync loop
function syncLoop() {
  if (!audioRef.current || !videoRef.current) return;

  const audioTime = audioRef.current.currentTime;
  const targetIdx = findActiveScene(audioTime);

  if (targetIdx !== currentSceneIdx) {
    // Scene changed — switch video source
    setCurrentSceneIdx(targetIdx);
    const boundary = sceneBoundaries[targetIdx];
    videoRef.current.src = boundary.clipUrl;

    // Seek video to the correct offset within the clip
    const clipOffset = audioTime - boundary.start;
    videoRef.current.currentTime = clipOffset;
    videoRef.current.play().catch(() => {});
  } else {
    // Same scene — correct drift if needed
    const boundary = sceneBoundaries[targetIdx];
    const expectedVideoTime = audioTime - boundary.start;
    const drift = Math.abs(videoRef.current.currentTime - expectedVideoTime);

    if (drift > 0.1) {
      // Drift > 100ms — hard seek to correct
      videoRef.current.currentTime = expectedVideoTime;
    }
  }

  rafRef.current = requestAnimationFrame(syncLoop);
}
```

### Video source switching

When switching scenes, the `<video>` element gets a new `src`. This causes a brief load delay (the browser must fetch and buffer the new clip). Two mitigation strategies:

**Option A: Preload next clip.** Keep a second hidden `<video>` element that preloads the next scene's clip. On scene switch, swap visibility. This eliminates the loading gap but doubles memory usage.

**Option B: Accept the gap.** Scene switches show a brief flash (1-2 frames) while the new clip loads. For QA purposes this is acceptable — the user knows they're watching a preview, not the final output.

**Recommended: Option A (dual video elements)** because lip sync QA requires continuous playback without interruption. The memory cost is trivial (two video elements of the same resolution).

```typescript
const videoARef = useRef<HTMLVideoElement>(null);
const videoBRef = useRef<HTMLVideoElement>(null);
const [activeVideo, setActiveVideo] = useState<"A" | "B">("A");

// When approaching a scene boundary (e.g., 1s before), preload next clip
// into the inactive video element
useEffect(() => {
  const nextIdx = currentSceneIdx + 1;
  if (nextIdx >= sceneBoundaries.length) return;

  const inactiveVideo = activeVideo === "A" ? videoBRef.current : videoARef.current;
  if (inactiveVideo) {
    inactiveVideo.src = sceneBoundaries[nextIdx].clipUrl;
    inactiveVideo.preload = "auto";
  }
}, [currentSceneIdx]);

// On scene switch, swap which video is visible
function switchToScene(idx: number) {
  const nextActive = activeVideo === "A" ? "B" : "A";
  const nextVideo = nextActive === "A" ? videoARef.current : videoBRef.current;
  // ... seek and play nextVideo ...
  setActiveVideo(nextActive);
}
```

### Drift correction

Video playback rate is fixed at 1x (no `playbackRate` manipulation). If drift exceeds 100ms, the video element is seeked to the correct position. This handles minor timing mismatches without visible artifacts.

For drift < 100ms, do nothing. Human perception of lip sync mismatch starts at ~80-100ms; below that it's acceptable.

---

## Transport Controls

### Seek bar

A single horizontal seek bar spanning the full song duration. The position is driven by `audio.currentTime`. Seeking the bar seeks the audio element, and the sync loop handles switching the video to the correct scene.

Render scene boundaries as tick marks on the seek bar so the user can see scene divisions.

### Play from scene

Clicking a scene in the storyboard (or a "Play" button on SceneRow) seeks audio to that scene's `time_start` and begins playback. This is the primary entry point — users will typically want to check a specific scene, not watch from the beginning.

```typescript
function playFromScene(sceneId: string) {
  const scene = scenes.find(s => s.id === sceneId);
  if (!scene || !audioRef.current) return;

  audioRef.current.currentTime = scene.time_start;
  audioRef.current.play();
  setIsPlaying(true);
  // syncLoop handles the rest
}
```

### Loop scene

Toggle: when enabled, audio loops between the current scene's `time_start` and `time_end`. Implemented by checking in the sync loop:

```typescript
if (loopScene && audioRef.current.currentTime >= loopEndTime) {
  audioRef.current.currentTime = loopStartTime;
}
```

This is essential for lip sync QA — the user watches a single scene on repeat to evaluate sync quality.

### Previous / Next scene

Skip buttons that seek to the previous/next scene boundary.

---

## Component Props

```typescript
interface PreviewPlayerProps {
  scenes: Scene[];
  audioUrl: string;           // relative path to original song audio
  initialSceneId?: string;    // start playback from this scene
  onClose: () => void;
}
```

## Integration in App.tsx

The preview player appears as an overlay or a panel above the storyboard. It's opened by:
- A "Preview" button in the pipeline bar (assembly step area)
- A "Play" button on individual scene rows
- Keyboard shortcut: `Space` when the storyboard is focused (toggles play/pause if player is open)

```tsx
{showPreview && (
  <PreviewPlayer
    scenes={scenes}
    audioUrl={state.config.song.audio_file}
    initialSceneId={previewStartScene}
    onClose={() => setShowPreview(false)}
  />
)}
```

---

## API: Audio Serving

The original song audio needs to be served with HTTP byte-range support for seeking. FastAPI's `FileResponse` supports this, but the current static file setup may not. Verify that the existing `/api/files/{path}` endpoint returns `Accept-Ranges: bytes` headers. If not, add:

```python
from starlette.responses import FileResponse

@app.get("/api/files/{file_path:path}")
async def serve_file(file_path: str):
    full_path = project_root / file_path
    if not full_path.exists():
        raise HTTPException(404)
    return FileResponse(full_path, media_type=_guess_media_type(full_path))
```

`FileResponse` handles range requests natively in Starlette.

---

## Scenes Without Clips

During preview playback, scenes with no video clip show:
- The scene's reference image (if available) as a static frame
- Or a dark placeholder with the scene ID and "No video generated"

The audio continues playing regardless. This lets the user preview a partially-complete project.

---

## Implementation Order

1. **PreviewPlayer component** — Dual-video element setup, audio master clock, sync loop with `requestAnimationFrame`, basic transport (play/pause/seek).

2. **Scene switching** — Preload next clip, swap video elements at boundaries, drift correction.

3. **Transport controls** — Seek bar with scene boundary ticks, play-from-scene, loop-scene, prev/next.

4. **Storyboard integration** — Play button on SceneRow, preview panel in App.tsx.

5. **Audio serving** — Verify byte-range support on the API file endpoint.

---

## What NOT to Do

- No video re-encoding for preview. Use the existing MP4 clips as-is.
- No assembled rough cut in the browser. The preview is a simulation using individual clips + original audio. The actual rough cut is for export.
- No transition preview. Transitions (crossfade, dip-to-black) are assembly-time effects and cannot be simulated cheaply in-browser. The preview shows hard cuts between scenes.
- No playback speed controls. 1x only. Slow-motion lip sync analysis can be done in DaVinci Resolve.
- No waveform in the preview player. The waveform editor is a separate tool for segmentation. The preview player is for QA.
