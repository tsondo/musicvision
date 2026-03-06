/**
 * WaveformEditor — Interactive scene boundary editor.
 *
 * Displays an audio waveform (wavesurfer.js v7) with:
 * - Draggable marker lines for scene boundaries
 * - AceStep section labels overlaid as colored regions
 * - Beat-snap toggle
 * - Integrated audio playback
 * - Scene list derived from markers
 * - Auto-segment fallback button
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import RegionsPlugin, { type Region } from "wavesurfer.js/dist/plugins/regions.js";
import type { AnalysisResult, SceneBoundary, SceneType, SongSection } from "../api/types";
import { fileUrl } from "../api/client";

interface Props {
  audioUrl: string; // relative path like "input/audio.wav"
  analysis: AnalysisResult;
  onConfirm: (boundaries: SceneBoundary[], snapToBeats: boolean) => void;
  onAutoSegment: (useLlm: boolean) => void;
  isRunning: boolean;
}

// Colors for section regions
const SECTION_COLORS: Record<string, string> = {
  intro: "rgba(91,138,245,0.12)",
  verse: "rgba(61,214,140,0.10)",
  chorus: "rgba(245,85,91,0.10)",
  hook: "rgba(245,85,91,0.10)",
  bridge: "rgba(245,197,66,0.10)",
  outro: "rgba(139,143,163,0.10)",
  pre: "rgba(170,120,255,0.10)",
};

function sectionColor(name: string): string {
  const lower = name.toLowerCase();
  for (const [key, color] of Object.entries(SECTION_COLORS)) {
    if (lower.includes(key)) return color;
  }
  return "rgba(91,138,245,0.08)";
}

function formatTime(s: number): string {
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return `${m}:${sec.toFixed(1).padStart(4, "0")}`;
}

export default function WaveformEditor({
  audioUrl,
  analysis,
  onConfirm,
  onAutoSegment,
  isRunning,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WaveSurfer | null>(null);
  const regionsRef = useRef<RegionsPlugin | null>(null);

  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [ready, setReady] = useState(false);
  const [markers, setMarkers] = useState<number[]>([]); // scene boundary times (not including 0 and duration)
  const [snapToBeats, setSnapToBeats] = useState(true);
  const [selectedMarker, setSelectedMarker] = useState<number | null>(null); // index in markers

  const duration = analysis.duration;
  const { beat_times, sections } = analysis;

  // Derive scenes from markers
  const derivedScenes = useMemo(() => {
    const points = [0, ...markers.sort((a, b) => a - b), duration];
    const scenes: { start: number; end: number; section: string; type: SceneType }[] = [];
    for (let i = 0; i < points.length - 1; i++) {
      const start = points[i];
      const end = points[i + 1];
      // Find section that falls within this range
      const sec = sections.find(
        (s) => s.time >= start && s.time < end,
      );
      scenes.push({
        start,
        end,
        section: sec?.name ?? "",
        type: "vocal",
      });
    }
    return scenes;
  }, [markers, duration, sections]);

  // Snap a time to nearest beat
  const snap = useCallback(
    (t: number): number => {
      if (!snapToBeats || beat_times.length === 0) return t;
      let closest = beat_times[0];
      let minDist = Math.abs(beat_times[0] - t);
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

  // Init wavesurfer
  useEffect(() => {
    if (!containerRef.current) return;

    const regions = RegionsPlugin.create();
    regionsRef.current = regions;

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
      plugins: [regions],
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

  // Add section label regions once ready
  useEffect(() => {
    if (!ready || !regionsRef.current) return;
    const reg = regionsRef.current;

    // Clear existing section regions (non-marker)
    for (const r of reg.getRegions()) {
      if (r.id.startsWith("section_")) r.remove();
    }

    // Add section regions
    for (let i = 0; i < sections.length; i++) {
      const sec = sections[i];
      const nextTime = i + 1 < sections.length ? sections[i + 1].time : duration;
      reg.addRegion({
        id: `section_${i}`,
        start: sec.time,
        end: nextTime,
        color: sectionColor(sec.name),
        drag: false,
        resize: false,
        content: sec.name,
      });
    }
  }, [ready, sections, duration]);

  // Sync marker regions with state
  useEffect(() => {
    if (!ready || !regionsRef.current) return;
    const reg = regionsRef.current;

    // Remove old marker regions
    for (const r of reg.getRegions()) {
      if (r.id.startsWith("marker_")) r.remove();
    }

    // Add markers as thin regions (visual dividers)
    for (let i = 0; i < markers.length; i++) {
      const t = markers[i];
      reg.addRegion({
        id: `marker_${i}`,
        start: t,
        end: t + 0.05, // near-zero width for a divider line
        color: "rgba(245, 197, 66, 0.7)",
        drag: true,
        resize: false,
      });
    }
  }, [ready, markers]);

  // Handle region drag (marker moved)
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

  // Add marker on waveform click (double-click)
  useEffect(() => {
    if (!wsRef.current || !ready) return;
    const ws = wsRef.current;

    const handleDblClick = () => {
      const t = snap(ws.getCurrentTime());
      // Don't add too close to existing markers
      const tooClose = markers.some((m) => Math.abs(m - t) < 0.5);
      if (tooClose || t < 0.5 || t > duration - 0.5) return;
      setMarkers((prev) => [...prev, Math.round(t * 100) / 100].sort((a, b) => a - b));
    };

    ws.on("dblclick", handleDblClick);
    return () => {
      ws.un("dblclick", handleDblClick);
    };
  }, [ready, markers, snap, duration]);

  const handlePlayPause = () => {
    wsRef.current?.playPause();
  };

  const handleDeleteMarker = (idx: number) => {
    setMarkers((prev) => prev.filter((_, i) => i !== idx));
    setSelectedMarker(null);
  };

  const handleClearAll = () => {
    setMarkers([]);
    setSelectedMarker(null);
  };

  // Pre-populate from sections
  const handleFromSections = () => {
    if (sections.length === 0) return;
    const newMarkers = sections
      .map((s) => snap(s.time))
      .filter((t) => t > 0.5 && t < duration - 0.5);
    setMarkers(newMarkers);
  };

  const handleConfirm = () => {
    const boundaries: SceneBoundary[] = derivedScenes.map((s) => ({
      time_start: s.start,
      time_end: s.end,
      section: s.section,
      type: s.type,
      lyrics: "",
    }));
    onConfirm(boundaries, snapToBeats);
  };

  const handleAutoSegment = () => {
    onAutoSegment(true);
  };

  return (
    <div className="waveform-editor">
      <div className="waveform-header">
        <h3>Scene Division</h3>
        <span className="waveform-hint">
          Double-click waveform to add markers. Drag markers to adjust. Scenes are created between markers.
        </span>
      </div>

      {/* Waveform */}
      <div className="waveform-container" ref={containerRef} />

      {/* Transport controls */}
      <div className="waveform-controls">
        <button
          className="btn btn-sm"
          onClick={handlePlayPause}
          disabled={!ready}
        >
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
          <button
            className="btn btn-sm"
            onClick={handleClearAll}
            disabled={markers.length === 0}
          >
            Clear All
          </button>
          <button
            className="btn btn-sm btn-secondary"
            onClick={handleAutoSegment}
            disabled={isRunning}
            title="Run LLM auto-segmentation"
          >
            {isRunning ? "Segmenting..." : "Auto-Segment"}
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

      {/* Scene list */}
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
                <th></th>
              </tr>
            </thead>
            <tbody>
              {derivedScenes.map((s, i) => (
                <tr
                  key={i}
                  className={selectedMarker === i ? "selected" : ""}
                  onClick={() => {
                    if (wsRef.current) wsRef.current.seekTo(s.start / duration);
                  }}
                >
                  <td>{i + 1}</td>
                  <td>{formatTime(s.start)}</td>
                  <td>{formatTime(s.end)}</td>
                  <td>{(s.end - s.start).toFixed(1)}s</td>
                  <td className="scene-section-cell">{s.section || "—"}</td>
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
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
