/**
 * WaveformEditor — Interactive timeline editor for scene boundary placement.
 *
 * Displays an audio waveform (wavesurfer.js v7) with:
 * - Zoom + minimap for navigation
 * - Section lane (colored blocks from AceStep metadata)
 * - Lyrics lane (word-level timestamps from Whisper, active word highlight)
 * - Gap indicators (safe cut points between words)
 * - Draggable marker lines for scene boundaries
 * - Scene duration warnings per engine constraints
 * - Beat-snap toggle
 * - Integrated audio playback
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import RegionsPlugin, { type Region } from "wavesurfer.js/dist/plugins/regions.js";
import Minimap from "wavesurfer.js/dist/plugins/minimap.js";
import type { AnalysisResult, SceneBoundary, SceneType, SongSection, VideoEngineType } from "../api/types";
import { fileUrl, getSegmentMarkers, saveSegmentMarkers } from "../api/client";

/* ------------------------------------------------------------------ */
/*  Props                                                              */
/* ------------------------------------------------------------------ */

interface Props {
  audioUrl: string; // relative path like "input/audio.wav"
  analysis: AnalysisResult;
  onConfirm: (boundaries: SceneBoundary[], snapToBeats: boolean) => void;
  onAutoSegment: (useLlm: boolean) => void;
  isRunning: boolean;
  videoEngine?: VideoEngineType;
  /** Externally-suggested markers (e.g. from auto-segment). Consumed once. */
  suggestedMarkers?: number[] | null;
  onClearScenes?: () => void;
}

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

/** Minimum inter-word silence to count as a gap (safe cut point). */
const GAP_THRESHOLD = 0.3; // seconds

/** Universal minimum scene duration (largest engine min: HVA 33 frames / 25fps). */
const MIN_SCENE_SECONDS = 1.32;

const ENGINE_CONSTRAINTS: Record<
  string,
  { name: string; maxFrames: number; minFrames: number; fps: number; maxSeconds: number; minSeconds: number }
> = {
  humo: { name: "HuMo", maxFrames: 97, minFrames: 25, fps: 25, maxSeconds: 3.88, minSeconds: 1.0 },
  hunyuan_avatar: { name: "HVA", maxFrames: 129, minFrames: 33, fps: 25, maxSeconds: 5.16, minSeconds: 1.32 },
  ltx_video: { name: "LTX-2", maxFrames: 257, minFrames: 9, fps: 24, maxSeconds: 10.71, minSeconds: 0.375 },
};

// Colors for section blocks (background tint)
const SECTION_COLORS: Record<string, string> = {
  intro: "rgba(91,138,245,0.18)",
  verse: "rgba(61,214,140,0.15)",
  chorus: "rgba(245,85,91,0.15)",
  hook: "rgba(245,85,91,0.15)",
  bridge: "rgba(245,197,66,0.15)",
  outro: "rgba(139,143,163,0.15)",
  pre: "rgba(170,120,255,0.15)",
};

// Text colors for lyrics words by section
const SECTION_TEXT_COLORS: Record<string, string> = {
  intro: "#5b8af5",
  verse: "#3dd68c",
  chorus: "#f5555b",
  hook: "#f5555b",
  bridge: "#f5c542",
  outro: "#8b8fa3",
  pre: "#aa78ff",
};

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

function sectionColor(name: string): string {
  const lower = name.toLowerCase();
  for (const [key, color] of Object.entries(SECTION_COLORS)) {
    if (lower.includes(key)) return color;
  }
  return "rgba(91,138,245,0.10)";
}

function sectionTextColor(sections: SongSection[], sectionIndex: number): string {
  const sec = sections[sectionIndex];
  if (sectionIndex < 0 || !sec) return "var(--text-dim)";
  const name = sec.name.toLowerCase();
  for (const [key, color] of Object.entries(SECTION_TEXT_COLORS)) {
    if (name.includes(key)) return color;
  }
  return "var(--text-dim)";
}

function formatTime(s: number): string {
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return `${m}:${sec.toFixed(1).padStart(4, "0")}`;
}

function assignSection(wordStart: number, sections: SongSection[]): number {
  for (let i = sections.length - 1; i >= 0; i--) {
    const sec = sections[i];
    if (sec && wordStart >= sec.time) return i;
  }
  return -1;
}

interface ProcessedWord {
  word: string;
  start: number;
  end: number;
  sectionIndex: number;
}

interface Gap {
  start: number;
  end: number;
  duration: number;
}

function sceneWarnings(sceneDuration: number, engineKey: string): string[] {
  const warnings: string[] = [];

  if (sceneDuration < MIN_SCENE_SECONDS) {
    warnings.push(`Too short (${sceneDuration.toFixed(1)}s < ${MIN_SCENE_SECONDS}s min)`);
    return warnings;
  }

  const c = ENGINE_CONSTRAINTS[engineKey];
  if (!c) return warnings;

  // Check sub-clip split using the same logic as backend compute_subclip_frames:
  // If naive ceil split leaves a remainder below min, try fewer clips.
  const totalFrames = Math.round(sceneDuration * c.fps);
  if (totalFrames > c.maxFrames) {
    let n = Math.ceil(totalFrames / c.maxFrames);
    const remainder = totalFrames - (n - 1) * c.maxFrames;
    if (remainder > 0 && remainder < c.minFrames && n > 1) {
      // Try reducing clip count (backend redistribution)
      const candidate = n - 1;
      if (candidate > 0 && Math.ceil(totalFrames / candidate) <= c.maxFrames) {
        n = candidate; // redistribution works — no warning needed
      } else {
        // Can't redistribute — genuinely problematic duration
        const remainderSec = (remainder / c.fps).toFixed(1);
        warnings.push(`Last sub-clip ${remainderSec}s (below ${c.name} min ${c.minSeconds.toFixed(1)}s)`);
      }
    }
  }

  return warnings;
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function WaveformEditor({
  audioUrl,
  analysis,
  onConfirm,
  onAutoSegment,
  isRunning,
  videoEngine,
  suggestedMarkers,
  onClearScenes,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const minimapRef = useRef<HTMLDivElement>(null);
  const timelineScrollRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WaveSurfer | null>(null);
  const regionsRef = useRef<RegionsPlugin | null>(null);

  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [ready, setReady] = useState(false);
  const [markers, setMarkers] = useState<number[]>([]);
  const [playingSceneIdx, setPlayingSceneIdx] = useState<number | null>(null);
  const sceneEndRef = useRef<number | null>(null);
  const [snapToBeats, setSnapToBeats] = useState(true);
  const [activeScene, setActiveScene] = useState<number | null>(null);
  const [selectedScenes, setSelectedScenes] = useState<Set<number>>(new Set());
  const [zoomLevel, setZoomLevel] = useState(0); // 0 = fit-to-width, else px/sec
  const [containerWidth, setContainerWidth] = useState(0);

  const duration = analysis.duration;
  const { beat_times, sections, word_timestamps } = analysis;
  const engineKey = videoEngine ?? "hunyuan_avatar";

  // ---- Load persisted markers on mount ----
  const loadedRef = useRef(false);
  useEffect(() => {
    if (loadedRef.current) return;
    loadedRef.current = true;
    getSegmentMarkers()
      .then((data) => {
        if (data.markers && data.markers.length > 0) {
          setMarkers(data.markers);
        }
      })
      .catch(() => {}); // no saved markers yet
  }, []);

  // ---- Consume suggested markers (from auto-segment) ----
  const prevSuggestedRef = useRef<number[] | null>(null);
  useEffect(() => {
    if (
      suggestedMarkers &&
      suggestedMarkers.length > 0 &&
      suggestedMarkers !== prevSuggestedRef.current
    ) {
      prevSuggestedRef.current = suggestedMarkers;
      setMarkers(suggestedMarkers);
    }
  }, [suggestedMarkers]);

  // ---- Auto-save markers on change (debounced) ----
  const saveTimer = useRef<ReturnType<typeof setTimeout>>(null);
  useEffect(() => {
    // Skip the initial empty state before load completes
    if (!loadedRef.current) return;
    if (saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => {
      saveSegmentMarkers(markers).catch(() => {});
    }, 500);
    return () => {
      if (saveTimer.current) clearTimeout(saveTimer.current);
    };
  }, [markers]);

  // ---- Container width measurement ----
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) setContainerWidth(entry.contentRect.width);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // ---- Time ↔ pixel conversion ----
  const timeToPixels = useCallback(
    (t: number) => {
      if (zoomLevel === 0) {
        // Account for wavesurfer container padding (4px each side)
        const w = Math.max(containerWidth - 8, 1);
        return (t / duration) * w;
      }
      return t * zoomLevel;
    },
    [zoomLevel, duration, containerWidth],
  );

  const totalWidth = useMemo(() => {
    if (zoomLevel === 0) return Math.max(containerWidth - 8, 1);
    return duration * zoomLevel;
  }, [zoomLevel, duration, containerWidth]);

  // ---- Processed words and gaps ----
  const processedWords = useMemo<ProcessedWord[]>(() => {
    return word_timestamps.map((w) => ({
      word: w.word,
      start: w.start,
      end: w.end,
      sectionIndex: assignSection(w.start, sections),
    }));
  }, [word_timestamps, sections]);

  const gaps = useMemo<Gap[]>(() => {
    const result: Gap[] = [];
    for (let i = 0; i < word_timestamps.length - 1; i++) {
      const curr = word_timestamps[i]!;
      const next = word_timestamps[i + 1]!;
      const gapStart = curr.end;
      const gapEnd = next.start;
      const dur = gapEnd - gapStart;
      if (dur >= GAP_THRESHOLD) {
        result.push({ start: gapStart, end: gapEnd, duration: dur });
      }
    }
    return result;
  }, [word_timestamps]);

  // ---- Active word (binary search) ----
  const activeWordIndex = useMemo(() => {
    const words = word_timestamps;
    if (words.length === 0) return -1;
    let lo = 0;
    let hi = words.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const w = words[mid]!;
      if (w.end <= currentTime) {
        lo = mid + 1;
      } else if (w.start > currentTime) {
        hi = mid - 1;
      } else {
        return mid;
      }
    }
    return -1;
  }, [currentTime, word_timestamps]);

  // ---- Derive scenes from markers ----
  const derivedScenes = useMemo(() => {
    const points = [0, ...markers.sort((a, b) => a - b), duration];
    const scenes: { start: number; end: number; section: string; type: SceneType }[] = [];
    for (let i = 0; i < points.length - 1; i++) {
      const start = points[i]!;
      const end = points[i + 1]!;
      const sec = sections.find((s) => s.time >= start && s.time < end);
      scenes.push({ start, end, section: sec?.name ?? "", type: "vocal" });
    }
    return scenes;
  }, [markers, duration, sections]);

  // ---- Snap to nearest beat ----
  const snap = useCallback(
    (t: number): number => {
      if (!snapToBeats || beat_times.length === 0) return t;
      let closest = beat_times[0]!;
      let minDist = Math.abs(closest - t);
      for (const bt of beat_times) {
        const d = Math.abs(bt - t);
        if (d < minDist) {
          minDist = d;
          closest = bt;
        }
      }
      return minDist < 0.3 ? closest : t;
    },
    [snapToBeats, beat_times],
  );

  // ---- Init wavesurfer ----
  useEffect(() => {
    if (!containerRef.current || !minimapRef.current) return;

    const regions = RegionsPlugin.create();
    regionsRef.current = regions;

    const minimap = Minimap.create({
      container: minimapRef.current,
      height: 32,
      waveColor: "#3dd68c44",
      progressColor: "#1a6b4644",
      cursorColor: "#f59e0b",
      normalize: true,
    });

    const ws = WaveSurfer.create({
      container: containerRef.current,
      height: 128,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      cursorWidth: 2,
      cursorColor: "#f59e0b",
      waveColor: "#3dd68c",
      progressColor: "#1a6b46",
      normalize: true,
      interact: true,
      plugins: [regions, minimap],
    });

    ws.load(fileUrl(audioUrl));

    ws.on("ready", () => setReady(true));
    ws.on("play", () => setPlaying(true));
    ws.on("pause", () => {
      setPlaying(false);
      setPlayingSceneIdx(null);
    });
    ws.on("timeupdate", (t) => {
      setCurrentTime(t);
      // Stop at scene end when previewing a single scene
      if (sceneEndRef.current !== null && t >= sceneEndRef.current) {
        ws.pause();
        sceneEndRef.current = null;
      }
    });

    wsRef.current = ws;

    return () => {
      ws.destroy();
      wsRef.current = null;
      regionsRef.current = null;
      setReady(false);
    };
  }, [audioUrl]);

  // ---- Zoom effect ----
  useEffect(() => {
    if (!wsRef.current || !ready) return;
    wsRef.current.zoom(zoomLevel);
  }, [zoomLevel, ready]);

  // ---- Ctrl+scroll zoom ----
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const handleWheel = (e: WheelEvent) => {
      if (!e.ctrlKey && !e.metaKey) return;
      e.preventDefault();
      setZoomLevel((prev) => {
        const delta = e.deltaY > 0 ? -10 : 10;
        return Math.max(0, Math.min(200, prev + delta));
      });
    };
    container.addEventListener("wheel", handleWheel, { passive: false });
    return () => container.removeEventListener("wheel", handleWheel);
  }, []);

  // ---- Scroll sync: wavesurfer → timeline lanes (unidirectional) ----
  useEffect(() => {
    if (!wsRef.current || !ready) return;
    const wrapper = wsRef.current.getWrapper();
    const timeline = timelineScrollRef.current;
    if (!wrapper || !timeline) return;

    const sync = () => {
      timeline.scrollLeft = wrapper.scrollLeft;
    };

    wrapper.addEventListener("scroll", sync);
    return () => wrapper.removeEventListener("scroll", sync);
  }, [ready]);

  // ---- Determine which marker indices are draggable (borders of active scene) ----
  const draggableMarkers = useMemo<Set<number>>(() => {
    if (activeScene === null) return new Set();
    const set = new Set<number>();
    // Scene i spans [markers[i-1], markers[i]]
    // Its start boundary is marker at index (activeScene - 1)
    // Its end boundary is marker at index (activeScene)
    if (activeScene > 0 && activeScene - 1 < markers.length) set.add(activeScene - 1);
    if (activeScene < markers.length) set.add(activeScene);
    return set;
  }, [activeScene, markers.length]);

  // ---- Sync marker regions + active scene highlight with state ----
  useEffect(() => {
    if (!ready || !regionsRef.current) return;
    const reg = regionsRef.current;

    for (const r of reg.getRegions()) {
      if (r.id.startsWith("marker_") || r.id === "active_scene") r.remove();
    }

    // Render inactive markers first, then active ones on top
    for (let i = 0; i < markers.length; i++) {
      if (draggableMarkers.has(i)) continue;
      reg.addRegion({
        id: `marker_${i}`,
        start: markers[i]!,
        end: markers[i]! + 0.05,
        color: "rgba(245, 197, 66, 0.35)",
        drag: false,
        resize: false,
      });
    }
    for (const i of draggableMarkers) {
      if (i < 0 || i >= markers.length) continue;
      reg.addRegion({
        id: `marker_${i}`,
        start: markers[i]!,
        end: markers[i]! + 0.05,
        color: "rgba(245, 158, 11, 0.95)",
        drag: true,
        resize: false,
      });
    }

    // Highlight the active scene region (non-interactive)
    if (activeScene !== null) {
      const sceneStart = activeScene > 0 ? (markers[activeScene - 1] ?? 0) : 0;
      const sceneEnd = activeScene < markers.length ? (markers[activeScene] ?? duration) : duration;
      reg.addRegion({
        id: "active_scene",
        start: sceneStart,
        end: sceneEnd,
        color: "rgba(245, 158, 11, 0.12)",
        drag: false,
        resize: false,
      });
    }
  }, [ready, markers, draggableMarkers, activeScene, duration]);

  // ---- Gap regions on waveform ----
  useEffect(() => {
    if (!ready || !regionsRef.current) return;
    const reg = regionsRef.current;

    for (const r of reg.getRegions()) {
      if (r.id.startsWith("gap_")) r.remove();
    }

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

  // ---- Handle marker drag — clamp to neighbors ----
  useEffect(() => {
    if (!regionsRef.current) return;
    const reg = regionsRef.current;

    const handleUpdate = (region: Region) => {
      if (!region.id.startsWith("marker_")) return;
      const idx = parseInt(region.id.replace("marker_", ""), 10);
      let newTime = snap(region.start);

      setMarkers((prev) => {
        const lo = idx > 0 ? (prev[idx - 1] ?? 0) : 0;
        const hi = idx < prev.length - 1 ? (prev[idx + 1] ?? duration) : duration;
        newTime = Math.max(lo, Math.min(hi, newTime));
        const next = [...prev];
        next[idx] = Math.round(newTime * 100) / 100;
        return next;
      });
    };

    reg.on("region-updated", handleUpdate);
    return () => {
      reg.un("region-updated", handleUpdate);
    };
  }, [snap, duration]);

  // ---- Double-click inserts a 1-second segment and selects it ----
  useEffect(() => {
    if (!wsRef.current || !ready) return;
    const ws = wsRef.current;

    const handleDblClick = () => {
      const t = snap(ws.getCurrentTime());
      // Need at least 1s from both edges
      if (t < 1.0 || t > duration - 1.0) return;

      const segStart = Math.round((t - 0.5) * 100) / 100;
      const segEnd = Math.round((t + 0.5) * 100) / 100;

      // Don't insert if too close to existing markers
      const tooClose = markers.some(
        (m) => Math.abs(m - segStart) < 0.1 || Math.abs(m - segEnd) < 0.1,
      );
      if (tooClose) return;

      setMarkers((prev) => {
        const next = [...prev, segStart, segEnd].sort((a, b) => a - b);
        // Find which scene index the new segment became
        const newIdx = next.indexOf(segStart) + 1;
        // Schedule selection after state update
        setTimeout(() => setActiveScene(newIdx), 0);
        return next;
      });
    };

    ws.on("dblclick", handleDblClick);
    return () => {
      ws.un("dblclick", handleDblClick);
    };
  }, [ready, markers, snap, duration]);

  // ---- Actions ----
  const handlePlayPause = () => wsRef.current?.playPause();

  const handleDeleteMarker = (markerIdx: number) => {
    // Removing a marker merges the two segments it separates
    setMarkers((prev) => prev.filter((_, i) => i !== markerIdx));
    setActiveScene(null);
    setSelectedScenes(new Set());
  };

  const handleClearAll = () => {
    setMarkers([]);
    setActiveScene(null);
    setSelectedScenes(new Set());
    onClearScenes?.();
  };

  const handleFromSections = () => {
    if (sections.length === 0) return;
    const newMarkers = sections
      .map((s) => snap(s.time))
      .filter((t) => t > 0.5 && t < duration - 0.5);
    setMarkers(newMarkers);
  };

  const handleConfirm = () => {
    // Populate lyrics from word timestamps (ground truth from Whisper)
    const getSceneLyrics = (start: number, end: number): string => {
      return analysis.word_timestamps
        .filter((w) => {
          const mid = (w.start + w.end) / 2;
          return mid >= start && mid <= end;
        })
        .map((w) => w.word)
        .join(" ");
    };

    const boundaries: SceneBoundary[] = derivedScenes.map((s) => ({
      time_start: s.start,
      time_end: s.end,
      section: s.section,
      type: s.type,
      lyrics: getSceneLyrics(s.start, s.end),
    }));
    onConfirm(boundaries, snapToBeats);
  };

  const toggleSceneSelection = (idx: number) => {
    setSelectedScenes((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  };

  const canMerge = useMemo(() => {
    if (selectedScenes.size < 2) return false;
    // Check that selected scenes are contiguous
    const sorted = [...selectedScenes].sort((a, b) => a - b);
    for (let i = 1; i < sorted.length; i++) {
      if (sorted[i]! - sorted[i - 1]! !== 1) return false;
    }
    return true;
  }, [selectedScenes]);

  const handleMerge = () => {
    if (!canMerge) return;
    const sorted = [...selectedScenes].sort((a, b) => a - b);
    // Markers to remove: between scene i and scene i+1 is marker at index i
    // (markers array is sorted, scene 0 = [0, markers[0]], scene 1 = [markers[0], markers[1]], etc.)
    // So merging scenes [first..last] means removing markers at indices [first..last-1]
    const first = sorted[0]!;
    const last = sorted[sorted.length - 1]!;
    const indicesToRemove = new Set<number>();
    for (let i = first; i < last; i++) {
      indicesToRemove.add(i);
    }
    setMarkers((prev) => prev.filter((_, i) => !indicesToRemove.has(i)));
    setSelectedScenes(new Set());
  };

  const handlePlayScene = (sceneIdx: number, start: number, end: number) => {
    const ws = wsRef.current;
    if (!ws) return;
    if (playingSceneIdx === sceneIdx) {
      // Already playing this scene — stop
      ws.pause();
      sceneEndRef.current = null;
      setPlayingSceneIdx(null);
      return;
    }
    sceneEndRef.current = end;
    setPlayingSceneIdx(sceneIdx);
    ws.seekTo(start / duration);
    ws.play();
  };

  const handleAutoSegment = () => {
    onAutoSegment(true);
  };

  // ---- Render ----
  return (
    <div className="waveform-editor">
      {/* Header with title + zoom */}
      <div className="waveform-header">
        <h3>Scene Timeline</h3>
        <span className="waveform-hint">
          Double-click waveform to add markers. Drag to adjust. Ctrl+scroll to zoom.
        </span>
        {analysis.warnings && analysis.warnings.length > 0 && (
          <span className="waveform-warning">
            {analysis.warnings.join(" | ")}
          </span>
        )}
        <div className="zoom-control">
          <span className="zoom-label">{zoomLevel === 0 ? "Fit" : `${zoomLevel} px/s`}</span>
          <input
            type="range"
            min={0}
            max={200}
            step={5}
            value={zoomLevel}
            onChange={(e) => setZoomLevel(Number(e.target.value))}
          />
        </div>
      </div>

      {/* Minimap — always full-width overview */}
      <div className="minimap-container" ref={minimapRef} />

      {/* Timeline lanes (section + lyrics) — scroll-synced with waveform */}
      <div
        className="timeline-scroll"
        ref={timelineScrollRef}
        style={{ overflowX: zoomLevel > 0 ? "auto" : "hidden", pointerEvents: "none" }}
      >
        {/* Section lane */}
        {sections.length > 0 && (
          <div className="section-lane" style={{ width: totalWidth }}>
            {sections.map((sec, i) => {
              const start = sec.time;
              const nextSec = sections[i + 1];
              const end = nextSec ? nextSec.time : duration;
              return (
                <div
                  key={i}
                  className="section-block"
                  style={{
                    left: timeToPixels(start),
                    width: timeToPixels(end - start),
                    backgroundColor: sectionColor(sec.name),
                  }}
                >
                  <span className="section-label">{sec.name}</span>
                </div>
              );
            })}
          </div>
        )}

        {/* Lyrics lane */}
        {processedWords.length > 0 && (
          <div className="lyrics-lane" style={{ width: totalWidth }}>
            {processedWords.map((pw, i) => (
              <span
                key={i}
                className={`lyric-word ${i === activeWordIndex ? "lyric-active" : ""}`}
                style={{
                  left: timeToPixels(pw.start),
                  color: sectionTextColor(sections, pw.sectionIndex),
                }}
                title={`${formatTime(pw.start)} – ${formatTime(pw.end)}`}
              >
                {pw.word}
              </span>
            ))}
            {gaps.map((g, i) => (
              <div
                key={`gap-${i}`}
                className="lyric-gap"
                style={{
                  left: timeToPixels(g.start),
                  width: Math.max(4, timeToPixels(g.duration)),
                }}
                title={`Gap: ${g.duration.toFixed(2)}s — safe cut point`}
              />
            ))}
          </div>
        )}
      </div>

      {/* Waveform */}
      <div className="waveform-container" ref={containerRef} />

      {/* Transport controls */}
      <div className="waveform-controls">
        <button className="btn btn-sm" onClick={handlePlayPause} disabled={!ready}>
          {playing ? "⏸ Pause" : "▶ Play"}
        </button>
        <span className="waveform-time">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>

        <label className="waveform-toggle">
          <input
            type="checkbox"
            checked={snapToBeats}
            onChange={(e) => setSnapToBeats(e.target.checked)}
          />
          Snap to beats
        </label>

        <button className="btn btn-sm" onClick={handleClearAll} disabled={markers.length === 0}>
          Clear All
        </button>
        {canMerge && (
          <button className="btn btn-sm" onClick={handleMerge} title="Merge selected adjacent segments">
            Merge {selectedScenes.size} Segments
          </button>
        )}
        <button
          className="btn btn-sm btn-secondary"
          onClick={handleAutoSegment}
          disabled={isRunning}
          title="Run LLM auto-segmentation and populate timeline"
        >
          {isRunning ? "Segmenting..." : "Suggest Segmentation"}
        </button>
        <button
          className="btn btn-sm btn-primary"
          onClick={handleConfirm}
          disabled={markers.length === 0 || isRunning}
        >
          Confirm {derivedScenes.length} Scenes
        </button>

        <div className="waveform-actions">
          {sections.length > 0 && (
            <button
              className="btn btn-sm"
              onClick={handleFromSections}
              disabled={isRunning}
              title="Place markers at section boundaries"
            >
              From Sections
            </button>
          )}
        </div>
      </div>

      {/* Scene table with warnings */}
      {derivedScenes.length > 0 && (
        <div className="waveform-scenes">
          <table className="scene-table">
            <thead>
              <tr>
                <th style={{ width: 28 }}></th>
                <th>#</th>
                <th></th>
                <th>Start</th>
                <th>End</th>
                <th>Duration</th>
                <th>Section</th>
                <th>Warnings</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {derivedScenes.map((s, i) => {
                const dur = s.end - s.start;
                const warnings = sceneWarnings(dur, engineKey);
                const isError = dur < MIN_SCENE_SECONDS;
                return (
                  <tr
                    key={i}
                    className={`${activeScene === i ? "selected" : ""} ${warnings.length > 0 ? "has-warning" : ""}`}
                    onClick={() => {
                      setActiveScene(activeScene === i ? null : i);
                      if (wsRef.current) wsRef.current.seekTo(s.start / duration);
                    }}
                  >
                    <td>
                      <input
                        type="checkbox"
                        checked={selectedScenes.has(i)}
                        onChange={(e) => {
                          e.stopPropagation();
                          toggleSceneSelection(i);
                        }}
                        onClick={(e) => e.stopPropagation()}
                      />
                    </td>
                    <td>{i + 1}</td>
                    <td>
                      <button
                        className="btn-icon btn-play-scene"
                        onClick={(e) => {
                          e.stopPropagation();
                          handlePlayScene(i, s.start, s.end);
                        }}
                        title={playingSceneIdx === i ? "Stop" : "Play this segment"}
                      >
                        {playingSceneIdx === i ? "\u25A0" : "\u25B6"}
                      </button>
                    </td>
                    <td>{formatTime(s.start)}</td>
                    <td>{formatTime(s.end)}</td>
                    <td>{dur.toFixed(1)}s</td>
                    <td className="scene-section-cell">{s.section || "—"}</td>
                    <td>
                      {warnings.map((w, wi) => (
                        <span key={wi} className={isError ? "scene-warning-error" : "scene-warning"}>
                          {w}
                        </span>
                      ))}
                    </td>
                    <td>
                      {i < markers.length && (
                        <button
                          className="btn-icon btn-delete-marker"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDeleteMarker(i);
                          }}
                          title="Remove this boundary"
                        >
                          ✕
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
