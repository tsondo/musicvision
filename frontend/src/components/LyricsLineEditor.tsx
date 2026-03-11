/**
 * LyricsMapper — Two-column lyrics-to-scene assignment.
 *
 * Left column: scene/segment list with play buttons. Click to select active scene.
 * Right column: lyrics lines with checkboxes. Checking a line assigns it to the
 * active scene. Section headers shown as separators, not assignable.
 *
 * Lines assigned to other scenes show a small badge with scene number(s).
 * "Try Whisper" button calls backend to auto-suggest assignments.
 */

import { useCallback, useState } from "react";
import { fileUrl } from "../api/client";

interface LyricsAssignment {
  line: string;
  scene_indices: number[];
  is_header?: boolean;
}

interface SceneInfo {
  start: number;
  end: number;
  section: string;
  audioSegment?: string | null;
}

interface Props {
  assignments: LyricsAssignment[];
  scenes: SceneInfo[];
  audioUrl: string;
  onChange: (assignments: LyricsAssignment[]) => void;
  onTryWhisper?: () => void;
  whisperRunning?: boolean;
}

const HEADER_PATTERN = /^\s*[\[\(].*[\]\)]\s*$/;

function formatTime(s: number): string {
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return `${m}:${sec.toFixed(0).padStart(2, "0")}`;
}

export default function LyricsMapper({
  assignments,
  scenes,
  audioUrl,
  onChange,
  onTryWhisper,
  whisperRunning,
}: Props) {
  const [activeScene, setActiveScene] = useState<number | null>(null);
  const [playingScene, setPlayingScene] = useState<number | null>(null);
  const [audioEl] = useState(() => new Audio());

  const toggleLine = useCallback(
    (lineIdx: number) => {
      if (activeScene === null) return;
      const updated = assignments.map((a, i) => {
        if (i !== lineIdx) return a;
        const has = a.scene_indices.includes(activeScene);
        return {
          ...a,
          scene_indices: has
            ? a.scene_indices.filter((s) => s !== activeScene)
            : [...a.scene_indices, activeScene].sort((a, b) => a - b),
        };
      });
      onChange(updated);
    },
    [assignments, activeScene, onChange],
  );

  const clearAll = useCallback(() => {
    onChange(assignments.map((a) => ({ ...a, scene_indices: [] })));
  }, [assignments, onChange]);

  const playScene = useCallback(
    (idx: number) => {
      const scene = scenes[idx];
      if (!scene) return;

      if (playingScene === idx) {
        audioEl.pause();
        setPlayingScene(null);
        return;
      }

      // Use scene audio segment if available, otherwise seek main audio
      const src = scene.audioSegment
        ? fileUrl(scene.audioSegment)
        : fileUrl(audioUrl);
      audioEl.src = src;
      if (!scene.audioSegment) {
        audioEl.currentTime = scene.start;
      }
      audioEl.play();
      setPlayingScene(idx);
      setActiveScene(idx);

      // Stop at scene end for main audio
      const handleTimeUpdate = () => {
        if (!scene.audioSegment && audioEl.currentTime >= scene.end) {
          audioEl.pause();
          audioEl.removeEventListener("timeupdate", handleTimeUpdate);
          setPlayingScene(null);
        }
      };
      const handleEnded = () => {
        setPlayingScene(null);
        audioEl.removeEventListener("ended", handleEnded);
      };
      audioEl.addEventListener("timeupdate", handleTimeUpdate);
      audioEl.addEventListener("ended", handleEnded);
    },
    [scenes, audioUrl, audioEl, playingScene],
  );

  if (assignments.length === 0) {
    return (
      <div className="lyrics-mapper">
        <div className="lyrics-editor-empty">No lyrics lines available. Import a lyrics file first.</div>
      </div>
    );
  }

  // Count lines assigned to active scene
  const activeCount = activeScene !== null
    ? assignments.filter((a) => a.scene_indices.includes(activeScene)).length
    : 0;

  return (
    <div className="lyrics-mapper">
      <div className="lyrics-mapper-header">
        <span className="lyrics-editor-title">Lyrics Mapper</span>
        {activeScene !== null && (
          <span className="lyrics-mapper-active-label">
            Scene {activeScene + 1} — {activeCount} line{activeCount !== 1 ? "s" : ""}
          </span>
        )}
        <div className="lyrics-mapper-actions">
          {onTryWhisper && (
            <button
              className="btn btn-sm"
              onClick={onTryWhisper}
              disabled={whisperRunning}
              title="Auto-detect lyrics for each scene using Whisper transcription"
            >
              {whisperRunning ? "Running..." : "Try Whisper"}
            </button>
          )}
          <button className="btn btn-sm" onClick={clearAll} title="Clear all assignments">
            Clear All
          </button>
        </div>
      </div>

      <div className="lyrics-mapper-columns">
        {/* Left: Scenes */}
        <div className="lyrics-mapper-scenes">
          {scenes.map((s, idx) => {
            const isActive = activeScene === idx;
            const isPlaying = playingScene === idx;
            const assignedLines = assignments.filter((a) => a.scene_indices.includes(idx) && !a.is_header && !HEADER_PATTERN.test(a.line));
            return (
              <div
                key={idx}
                className={`lyrics-mapper-scene ${isActive ? "active" : ""}`}
                onClick={() => setActiveScene(idx)}
              >
                <button
                  className="btn-icon btn-play-scene"
                  onClick={(e) => { e.stopPropagation(); playScene(idx); }}
                  title={isPlaying ? "Stop" : "Play"}
                >
                  {isPlaying ? "\u25A0" : "\u25B6"}
                </button>
                <span className="lyrics-mapper-scene-num">{idx + 1}</span>
                <span className="lyrics-mapper-scene-time">
                  {formatTime(s.start)}–{formatTime(s.end)}
                </span>
                {s.section && <span className="lyrics-mapper-scene-section">{s.section}</span>}
                {assignedLines.length > 0 && (
                  <span className="lyrics-mapper-scene-count">{assignedLines.length} lines</span>
                )}
              </div>
            );
          })}
        </div>

        {/* Right: Lyrics */}
        <div className="lyrics-mapper-lyrics">
          {activeScene === null ? (
            <div className="lyrics-mapper-hint">Select a scene on the left to assign lyrics</div>
          ) : (
            assignments.map((a, lineIdx) => {
              const isHeader = a.is_header || HEADER_PATTERN.test(a.line);
              if (isHeader) {
                return (
                  <div key={lineIdx} className="lyrics-mapper-section-header">
                    {a.line}
                  </div>
                );
              }

              const isChecked = a.scene_indices.includes(activeScene);
              const otherScenes = a.scene_indices.filter((s) => s !== activeScene);

              return (
                <label key={lineIdx} className={`lyrics-mapper-line ${isChecked ? "checked" : ""}`}>
                  <input
                    type="checkbox"
                    checked={isChecked}
                    onChange={() => toggleLine(lineIdx)}
                  />
                  <span className="lyrics-mapper-line-text">{a.line}</span>
                  {otherScenes.length > 0 && (
                    <span className="lyrics-mapper-other-badges">
                      {otherScenes.map((s) => (
                        <span key={s} className="lyrics-mapper-badge">{s + 1}</span>
                      ))}
                    </span>
                  )}
                </label>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}
