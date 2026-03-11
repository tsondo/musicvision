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
import { fileUrl } from "../api/client";

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

  // Check sub-clip remainder
  const totalFrames = Math.round(sceneDuration * c.fps);
  if (totalFrames > c.maxFrames) {
    const nClips = Math.ceil(totalFrames / c.maxFrames);
    const remainder = totalFrames - (nClips - 1) * c.maxFrames;
    if (remainder > 0 && remainder < c.minFrames) {
      const remainderSec = (remainder / c.fps).toFixed(1);
      warnings.push(`Last sub-clip ${remainderSec}s (below ${c.name} min ${c.minSeconds.toFixed(1)}s)`);
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
  const [snapToBeats, setSnapToBeats] = useState(true);
  const [selectedMarker, setSelectedMarker] = useState<number | null>(null);
  const [zoomLevel, setZoomLevel] = useState(0); // 0 = fit-to-width, else px/sec
  const [containerWidth, setContainerWidth] = useState(0);

  const duration = analysis.duration;
  const { beat_times, sections, word_timestamps } = analysis;
  const engineKey = videoEngine ?? "hunyuan_avatar";

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
    ws.on("pause", () => setPlaying(false));
    ws.on("timeupdate", (t) => setCurrentTime(t));

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

  // ---- Sync marker regions with state ----
  useEffect(() => {
    if (!ready || !regionsRef.current) return;
    const reg = regionsRef.current;

    for (const r of reg.getRegions()) {
      if (r.id.startsWith("marker_")) r.remove();
    }

    for (let i = 0; i < markers.length; i++) {
      const t = markers[i]!;
      reg.addRegion({
        id: `marker_${i}`,
        start: t,
        end: t + 0.05,
        color: "rgba(245, 197, 66, 0.7)",
        drag: true,
        resize: false,
      });
    }
  }, [ready, markers]);

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

  // ---- Handle region drag (marker moved) ----
  useEffect(() => {
    if (!regionsRef.current) return;
    const reg = regionsRef.current;

    const handleUpdate = (region: Region) => {
      if (!region.id.startsWith("marker_")) return;
      const idx = parseInt(region.id.replace("marker_", ""), 10);
      const newTime = snap(region.start);
      setMarkers((prev) => {
        const next = [...prev];
        next[idx] = Math.round(newTime * 100) / 100;
        return next;
      });
    };

    reg.on("region-updated", handleUpdate);
    return () => {
      reg.un("region-updated", handleUpdate);
    };
  }, [snap]);

  // ---- Add marker on double-click ----
  useEffect(() => {
    if (!wsRef.current || !ready) return;
    const ws = wsRef.current;

    const handleDblClick = () => {
      const t = snap(ws.getCurrentTime());
      const tooClose = markers.some((m) => Math.abs(m - t) < 0.5);
      if (tooClose || t < 0.5 || t > duration - 0.5) return;
      setMarkers((prev) => [...prev, Math.round(t * 100) / 100].sort((a, b) => a - b));
    };

    ws.on("dblclick", handleDblClick);
    return () => {
      ws.un("dblclick", handleDblClick);
    };
  }, [ready, markers, snap, duration]);

  // ---- Actions ----
  const handlePlayPause = () => wsRef.current?.playPause();

  const handleDeleteMarker = (idx: number) => {
    setMarkers((prev) => prev.filter((_, i) => i !== idx));
    setSelectedMarker(null);
  };

  const handleClearAll = () => {
    setMarkers([]);
    setSelectedMarker(null);
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

        <div className="waveform-actions">
          {sections.length > 0 && (
            <button
              className="btn btn-sm"
              onClick={handleFromSections}
              disabled={isRunning}
              title="Place markers at section boundaries from AceStep"
            >
              From Sections
            </button>
          )}
          <button className="btn btn-sm" onClick={handleClearAll} disabled={markers.length === 0}>
            Clear All
          </button>
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
        </div>
      </div>

      {/* Scene table with warnings */}
      {derivedScenes.length > 0 && (
        <div className="waveform-scenes">
          <table className="scene-table">
            <thead>
              <tr>
                <th>#</th>
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
                    className={`${selectedMarker === i ? "selected" : ""} ${warnings.length > 0 ? "has-warning" : ""}`}
                    onClick={() => {
                      if (wsRef.current) wsRef.current.seekTo(s.start / duration);
                    }}
                  >
                    <td>{i + 1}</td>
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
