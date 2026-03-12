# Timeline Editor Enhancement Spec

**Target file:** `frontend/src/components/WaveformEditor.tsx`
**Supporting CSS:** `frontend/src/App.css` (waveform-editor section)
**Integration:** `frontend/src/App.tsx` (workflow changes)

## Goal

Enhance the WaveformEditor into a fully interactive timeline editor with zoom, synced lyrics display, section markers, and gap detection. The primary purpose is letting the user place scene boundaries at musically and lyrically appropriate moments — specifically at non-spoken gaps — before the storyboard phase begins.

**Priority: Accuracy of lyric-to-audio sync is the most important requirement. Style is secondary to function.**

---

## Data Available

The `AnalysisResult` prop already provides everything needed:

```typescript
interface AnalysisResult {
  duration: number;                    // total seconds
  bpm: number | null;
  beat_times: number[];                // every beat, in seconds
  word_timestamps: Array<{             // from Whisper large-v3
    word: string;
    start: number;                     // seconds
    end: number;                       // seconds
  }>;
  vocal_path: string | null;
  sections: Array<{                    // from AceStep metadata
    name: string;                      // e.g. "Verse 1", "Chorus"
    time: number;                      // start time in seconds
  }>;
}
```

## Engine Constraints (for validation display)

Import from the API types or hardcode the current values for display purposes:

```
humo:            max 97 frames @ 25fps  = 3.88s max,  1.00s min
ltx_video:       max 257 frames @ 24fps = 10.71s max, 0.375s min
```

The editor does NOT enforce these as hard limits. It shows warnings. Sub-clip splitting is handled downstream by `engine_registry.py`. The editor's job is to let the user make informed decisions about where to place scene boundaries.

---

## Layout

The timeline editor is a single vertical stack. All horizontal elements share the same time axis and scroll together.

```
┌──────────────────────────────────────────────────────────────┐
│ [Header: "Scene Timeline"]  [Zoom slider ─────○──]           │
│                              [Hint text]                     │
├──────────────────────────────────────────────────────────────┤
│ MINIMAP — full song, fixed width, viewport rectangle         │
├──────────────────────────────────────────────────────────────┤
│ SECTION LANE — colored blocks with section labels            │
│ ┌─ Intro ──┬── Verse 1 ─────────┬── Chorus ────────┬─ ...   │
├──────────────────────────────────────────────────────────────┤
│ LYRICS LANE — word-level text, positioned by timestamp       │
│  ♪ Hello  world  │  goodbye  moon ♪    ♪ sing  along ♪      │
│         ▲ gap marker (safe cut)                              │
├──────────────────────────────────────────────────────────────┤
│ WAVEFORM — wavesurfer.js main display                        │
│ ┃▓▓▓▓▓▓▓│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│▓▓▓▓▓       │
│         ↑ draggable marker                                   │
├──────────────────────────────────────────────────────────────┤
│ TRANSPORT: [▶ Play] 1:23.4 / 3:45.0  [☐ Snap to beats]     │
│ ACTIONS:  [From Sections] [Suggest Segmentation] [Clear All] │
│           [Confirm N Scenes]                                  │
├──────────────────────────────────────────────────────────────┤
│ SCENE TABLE — derived scenes list (existing, unchanged)       │
│ # | Start | End | Duration | Section | Warnings | ✕          │
└──────────────────────────────────────────────────────────────┘
```

### Scroll synchronization

The section lane, lyrics lane, and waveform container MUST scroll horizontally in sync. Implementation: wrap all three in a single `overflow-x: auto` scroll container, or use a shared `scrollLeft` sync via refs. The minimap stays outside this scroll container (fixed width).

---

## Feature 1: Zoom

### Wavesurfer zoom

Wavesurfer.js v7 supports `ws.zoom(pixelsPerSecond)`. The default (no zoom) renders the entire track in the container width.

**State:**
```typescript
const [zoomLevel, setZoomLevel] = useState(0); // 0 = fit-to-width
```

**Zoom range:**
- `0` = fit to container (wavesurfer default, ~10-20 px/sec for a 3-4 min song)
- Slider range: 0 to 200 px/sec
- Step: 5 px/sec for the slider; finer control via keyboard/scroll

**Controls:**
- Horizontal slider in the header row, right-aligned
- Label: current zoom level or "Fit" when at 0
- Keyboard: `+` / `-` keys (increment by 10 px/sec), but only when the editor is focused (not when typing in an input)
- Ctrl+scroll (or Cmd+scroll on Mac) on the waveform container: zoom in/out centered on cursor position

**Implementation:**
```typescript
useEffect(() => {
  if (!wsRef.current || !ready) return;
  if (zoomLevel === 0) {
    // Reset to fit-to-width: set minPxPerSec to 0
    wsRef.current.zoom(0);
  } else {
    wsRef.current.zoom(zoomLevel);
  }
}, [zoomLevel, ready]);
```

When zoomed, the waveform container becomes horizontally scrollable. Wavesurfer handles this natively — the container gets `overflow-x: auto` and the internal canvas stretches.

### Minimap

Use wavesurfer's Minimap plugin:

```typescript
import Minimap from "wavesurfer.js/dist/plugins/minimap.js";
```

Add the minimap plugin during WaveSurfer.create():

```typescript
const minimap = Minimap.create({
  container: minimapRef.current,  // dedicated div above the main waveform
  height: 32,
  waveColor: "#3dd68c44",
  progressColor: "#1a6b4644",
  cursorColor: "#f59e0b",
  normalize: true,
});

const ws = WaveSurfer.create({
  // ... existing config ...
  plugins: [regions, minimap],
});
```

The minimap always shows the full track. When zoomed, it displays a semi-transparent viewport rectangle indicating the visible portion. Clicking the minimap seeks to that position.

**Minimap ref:**
```typescript
const minimapRef = useRef<HTMLDivElement>(null);
```

Render the minimap div ABOVE the section/lyrics/waveform scroll area.

---

## Feature 2: Lyrics Lane

### Data preprocessing

Compute these once from `analysis.word_timestamps` and `analysis.sections`:

```typescript
interface ProcessedWord {
  word: string;
  start: number;      // seconds
  end: number;        // seconds
  sectionIndex: number;  // index into sections array (-1 if before first section)
}

interface Gap {
  start: number;      // end of previous word
  end: number;        // start of next word
  duration: number;   // seconds
}
```

**Gap computation:**
```typescript
const gaps: Gap[] = useMemo(() => {
  const result: Gap[] = [];
  const words = analysis.word_timestamps;
  for (let i = 0; i < words.length - 1; i++) {
    const gapStart = words[i].end;
    const gapEnd = words[i + 1].start;
    const dur = gapEnd - gapStart;
    if (dur >= GAP_THRESHOLD) {
      result.push({ start: gapStart, end: gapEnd, duration: dur });
    }
  }
  return result;
}, [analysis.word_timestamps]);
```

**`GAP_THRESHOLD`:** 0.3 seconds. Gaps shorter than this are normal inter-word pauses and not useful as scene boundaries. Expose as a constant at the top of the file, not a user control.

### Section assignment

Each word gets a `sectionIndex` based on which section it falls within:

```typescript
function assignSection(wordStart: number, sections: SongSection[]): number {
  for (let i = sections.length - 1; i >= 0; i--) {
    if (wordStart >= sections[i].time) return i;
  }
  return -1; // before first section
}
```

### Rendering

The lyrics lane is a `<div>` with `position: relative` whose width matches the zoomed waveform width. Each word is an absolutely-positioned `<span>`:

```typescript
<div className="lyrics-lane" style={{ width: totalWidth }}>
  {processedWords.map((pw, i) => (
    <span
      key={i}
      className={`lyric-word ${pw.start <= currentTime && currentTime < pw.end ? "lyric-active" : ""}`}
      style={{
        position: "absolute",
        left: `${timeToPixels(pw.start)}px`,
        color: sectionTextColor(pw.sectionIndex),
      }}
      title={`${formatTime(pw.start)} – ${formatTime(pw.end)}`}
    >
      {pw.word}
    </span>
  ))}
  {/* Gap indicators */}
  {gaps.map((g, i) => (
    <div
      key={`gap-${i}`}
      className="lyric-gap"
      style={{
        position: "absolute",
        left: `${timeToPixels(g.start)}px`,
        width: `${timeToPixels(g.duration)}px`,
      }}
      title={`Gap: ${g.duration.toFixed(2)}s — safe cut point`}
    />
  ))}
</div>
```

**`timeToPixels` function:**
```typescript
function timeToPixels(t: number): number {
  if (zoomLevel === 0) {
    // Fit-to-width: use container width / duration
    return (t / duration) * containerWidth;
  }
  return t * zoomLevel; // zoomLevel is px/sec
}
```

`containerWidth` is measured from the scroll container's `clientWidth` via a ResizeObserver or ref measurement.

### Active word highlighting

The `lyric-active` class is applied when `pw.start <= currentTime < pw.end`. This updates on every `timeupdate` event from wavesurfer (which fires frequently during playback). The CSS for `lyric-active`:

```css
.lyric-word {
  font-size: 11px;
  font-family: var(--font);
  white-space: nowrap;
  padding: 1px 2px;
  border-radius: 2px;
  transition: background 0.05s;
  cursor: default;
  user-select: none;
}

.lyric-active {
  background: var(--accent);
  color: #fff;
  font-weight: 600;
}
```

**Performance note:** With hundreds of words, the active-word check runs on every `timeupdate`. Use a binary search on sorted `start` times to find the active word in O(log n) rather than iterating all words:

```typescript
const activeWordIndex = useMemo(() => {
  // Binary search for the word whose start <= currentTime < end
  const words = analysis.word_timestamps;
  let lo = 0, hi = words.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (words[mid].end <= currentTime) {
      lo = mid + 1;
    } else if (words[mid].start > currentTime) {
      hi = mid - 1;
    } else {
      return mid;
    }
  }
  return -1;
}, [currentTime, analysis.word_timestamps]);
```

Then apply the class based on index match rather than per-word time comparison in the render loop.

---

## Feature 3: Section Lane

The section lane renders colored blocks for each section, horizontally aligned with the waveform and lyrics. This replaces the wavesurfer region-based section rendering (remove the `section_` regions from the RegionsPlugin).

```typescript
<div className="section-lane" style={{ width: totalWidth }}>
  {sections.map((sec, i) => {
    const start = sec.time;
    const end = i + 1 < sections.length ? sections[i + 1].time : duration;
    return (
      <div
        key={i}
        className="section-block"
        style={{
          position: "absolute",
          left: `${timeToPixels(start)}px`,
          width: `${timeToPixels(end - start)}px`,
          backgroundColor: sectionColor(sec.name),
        }}
      >
        <span className="section-label">{sec.name}</span>
      </div>
    );
  })}
</div>
```

**CSS:**
```css
.section-lane {
  position: relative;
  height: 24px;
  border-bottom: 1px solid var(--border);
}

.section-block {
  position: absolute;
  top: 0;
  height: 100%;
  display: flex;
  align-items: center;
  padding-left: 6px;
  border-right: 1px solid var(--border);
  overflow: hidden;
}

.section-label {
  font-size: 10px;
  font-weight: 600;
  color: var(--text-dim);
  white-space: nowrap;
  text-overflow: ellipsis;
  overflow: hidden;
}
```

### Color coding

Use the existing `SECTION_COLORS` map for section block backgrounds. For lyrics word coloring, use a higher-opacity text color variant derived from the same palette:

```typescript
const SECTION_TEXT_COLORS: Record<string, string> = {
  intro:  "#5b8af5",
  verse:  "#3dd68c",
  chorus: "#f5555b",
  hook:   "#f5555b",
  bridge: "#f5c542",
  outro:  "#8b8fa3",
  pre:    "#aa78ff",
};

function sectionTextColor(sectionIndex: number): string {
  if (sectionIndex < 0 || sectionIndex >= sections.length) return "var(--text-dim)";
  const name = sections[sectionIndex].name.toLowerCase();
  for (const [key, color] of Object.entries(SECTION_TEXT_COLORS)) {
    if (name.includes(key)) return color;
  }
  return "var(--text-dim)";
}
```

---

## Feature 4: Gap Indicators

Gaps (inter-word silences ≥ 0.3s) are rendered in BOTH the lyrics lane and on the waveform.

### Lyrics lane gap markers

Already shown above — `<div className="lyric-gap">` elements.

```css
.lyric-gap {
  position: absolute;
  top: 50%;
  height: 2px;
  background: rgba(245, 197, 66, 0.4);
  border-radius: 1px;
  min-width: 4px;
  pointer-events: none;
}
```

### Waveform gap overlay

Render gap indicators as wavesurfer regions (non-draggable, non-resizable) with a distinct visual style:

```typescript
useEffect(() => {
  if (!ready || !regionsRef.current) return;
  const reg = regionsRef.current;

  // Remove old gap regions
  for (const r of reg.getRegions()) {
    if (r.id.startsWith("gap_")) r.remove();
  }

  // Add gap regions
  gaps.forEach((g, i) => {
    reg.addRegion({
      id: `gap_${i}`,
      start: g.start,
      end: g.end,
      color: "rgba(245, 197, 66, 0.15)",
      drag: false,
      resize: false,
    });
  });
}, [ready, gaps]);
```

Gaps should NOT have text labels — they'd clutter the waveform. The yellow tint is sufficient to indicate "safe cut zones." The user can hover for a tooltip (wavesurfer region hover events or the `title` attribute approach).

---

## Feature 5: Scene Duration Warnings

In the existing scene table (`derivedScenes` list at the bottom), add a warnings column.

### Warning logic

```typescript
function sceneWarnings(
  sceneDuration: number,
  engineKey: string,  // from a selector or project config
): string[] {
  const warnings: string[] = [];

  // Hard minimum: 1 second for any engine
  if (sceneDuration < 1.0) {
    warnings.push("Too short (< 1s)");
    return warnings;
  }

  // Engine-specific: check if duration creates a problematic sub-clip remainder
  const constraints = ENGINE_CONSTRAINTS[engineKey];
  if (constraints) {
    if (sceneDuration < constraints.minSeconds) {
      warnings.push(`Below ${constraints.name} minimum (${constraints.minSeconds.toFixed(1)}s)`);
    }

    // Check sub-clip remainder
    const totalFrames = Math.round(sceneDuration * constraints.fps);
    if (totalFrames > constraints.maxFrames) {
      const remainder = totalFrames % constraints.maxFrames;
      if (remainder > 0 && remainder < constraints.minFrames) {
        const remainderSec = (remainder / constraints.fps).toFixed(1);
        warnings.push(`Last sub-clip would be ${remainderSec}s (below engine minimum)`);
      }
    }
  }

  return warnings;
}
```

### Engine constraints (frontend constants)

Add to the component or a shared constants file:

```typescript
const ENGINE_CONSTRAINTS: Record<string, {
  name: string;
  maxFrames: number;
  minFrames: number;
  fps: number;
  maxSeconds: number;
  minSeconds: number;
}> = {
  humo: { name: "HuMo", maxFrames: 97, minFrames: 25, fps: 25, maxSeconds: 3.88, minSeconds: 1.0 },
  ltx_video: { name: "LTX-2", maxFrames: 257, minFrames: 9, fps: 24, maxSeconds: 10.71, minSeconds: 0.375 },
};
```

### Display

In the scene table, add a "Warnings" column. Cells with warnings get yellow text. Cells with hard errors (below absolute minimum) get red.

Also: add a subtle red/orange left-border on the scene table row when there are warnings, so they're visible even when the table is compact.

---

## Feature 6: Workflow Integration

### Auto-open after analysis

In `App.tsx`, when `pipeline.analyzeStatus` transitions to `"done"`, automatically show the waveform editor:

```typescript
useEffect(() => {
  if (pipeline.analyzeStatus === "done" && pipeline.analysisResult) {
    setShowWaveformEditor(true);
  }
}, [pipeline.analyzeStatus, pipeline.analysisResult]);
```

### Don't auto-close on confirm

Currently, `onConfirm` in App.tsx calls `setShowWaveformEditor(false)`. Keep this behavior — after confirming scenes, the storyboard appears below and the editor closes. The user can reopen it from the pipeline bar if they want to re-segment.

---

## Implementation Order

Execute in this order. Each step is independently testable.

1. **Zoom + Minimap** — Add zoom state, slider, minimap plugin. Verify horizontal scroll works when zoomed. This changes the waveform container behavior and must be done first since lyrics/sections depend on `timeToPixels`.

2. **Section lane** — Extract section rendering from wavesurfer regions into a DOM-based lane above the waveform. Remove `section_` regions from the RegionsPlugin. Verify horizontal scroll sync.

3. **Lyrics lane** — Add word positioning, section color coding, active word highlighting. Verify sync accuracy during playback.

4. **Gap detection + indicators** — Compute gaps, render in lyrics lane and as waveform regions. Verify alignment with word timestamps.

5. **Scene duration warnings** — Add engine constraints constants, warning logic, and table column.

6. **Workflow integration** — Auto-open after analysis.

---

## What NOT to Change

- **Marker interaction** — double-click to add, drag to move, table click to seek. All working, don't touch.
- **Beat snap** — existing `snap()` function is correct.
- **"From Sections" / "Clear All" / "Auto-Segment"** buttons — keep as-is.
- **`onConfirm` / `onAutoSegment` callbacks** — signatures unchanged.
- **Scene table** — extend (add warnings column), don't restructure.
- **Backend** — no API changes needed. All data is already in `AnalysisResult`.
- **Transport controls** — play/pause, time display, snap toggle stay where they are.

---

## Testing

### Manual verification checklist

- [ ] At zoom = 0, lyrics and sections align with waveform (play and watch active word sync)
- [ ] At zoom = 100px/sec, scroll horizontally — lyrics, sections, and waveform stay aligned
- [ ] Minimap shows viewport rectangle when zoomed, clicking minimap navigates
- [ ] Gap indicators appear at silent moments between words (verify by ear)
- [ ] Double-clicking on a gap indicator area creates a marker at that gap (existing behavior, should still work)
- [ ] Scene table warnings appear for scenes with problematic durations
- [ ] Ctrl+scroll zooms in/out smoothly
- [ ] Performance is acceptable with 200+ words (typical for a 3-4 minute song)
- [ ] Editor auto-opens after analysis completes

### Edge cases

- Song with no lyrics / no word timestamps → lyrics lane is empty, no gaps. Section lane and waveform still work.
- Song with no AceStep sections → section lane is empty. Lyrics lane words all get default color.
- Song with no beats → snap toggle disabled (existing behavior), zoom still works.
- Very short song (< 30s) → zoom range should still be usable; minimap may be unnecessary but shouldn't break.
- Very long song (> 10 min) → verify scroll performance at high zoom levels.
