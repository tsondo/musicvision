/**
 * LyricsLineEditor — Manual assignment of lyrics lines to scenes.
 *
 * Shows each lyrics line with toggle buttons to assign it to one or more
 * scenes. Section headers (e.g. [intro], (Verse 1)) are shown as visual
 * separators but are not assignable. Scene column headers show time ranges.
 */

import { useCallback } from "react";

interface LyricsAssignment {
  line: string;
  scene_indices: number[];
  is_header?: boolean;
}

interface SceneInfo {
  start: number;
  end: number;
  section: string;
}

interface Props {
  assignments: LyricsAssignment[];
  scenes: SceneInfo[];
  onChange: (assignments: LyricsAssignment[]) => void;
}

const HEADER_PATTERN = /^\s*[\[\(].*[\]\)]\s*$/;

function formatTime(s: number): string {
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return `${m}:${sec.toFixed(0).padStart(2, "0")}`;
}

export default function LyricsLineEditor({ assignments, scenes, onChange }: Props) {
  const sceneCount = scenes.length;

  const toggleScene = useCallback(
    (lineIdx: number, sceneIdx: number) => {
      const updated = assignments.map((a, i) => {
        if (i !== lineIdx) return a;
        const has = a.scene_indices.includes(sceneIdx);
        return {
          ...a,
          scene_indices: has
            ? a.scene_indices.filter((s) => s !== sceneIdx)
            : [...a.scene_indices, sceneIdx].sort((a, b) => a - b),
        };
      });
      onChange(updated);
    },
    [assignments, onChange],
  );

  const autoPopulate = useCallback(() => {
    const assignable = assignments.filter((a) => !a.is_header && !HEADER_PATTERN.test(a.line));
    if (assignable.length === 0 || sceneCount === 0) return;
    const linesPerScene = Math.max(1, Math.ceil(assignable.length / sceneCount));
    let assignableIdx = 0;
    const updated = assignments.map((a) => {
      if (a.is_header || HEADER_PATTERN.test(a.line)) {
        return { ...a, scene_indices: [] };
      }
      const sceneIdx = Math.min(Math.floor(assignableIdx / linesPerScene), sceneCount - 1);
      assignableIdx++;
      return { ...a, scene_indices: [sceneIdx] };
    });
    onChange(updated);
  }, [assignments, sceneCount, onChange]);

  const clearAll = useCallback(() => {
    onChange(assignments.map((a) => ({ ...a, scene_indices: [] })));
  }, [assignments, onChange]);

  if (assignments.length === 0) {
    return (
      <div className="lyrics-line-editor">
        <div className="lyrics-editor-empty">No lyrics lines available. Import a lyrics file first.</div>
      </div>
    );
  }

  const sceneOptions = Array.from({ length: sceneCount }, (_, i) => i);

  return (
    <div className="lyrics-line-editor">
      <div className="lyrics-editor-header">
        <span className="lyrics-editor-title">Lyrics → Scene Assignment</span>
        <button className="btn btn-sm" onClick={autoPopulate} title="Auto-distribute lines evenly across scenes">
          Auto-populate
        </button>
        <button className="btn btn-sm" onClick={clearAll} title="Clear all assignments">
          Clear
        </button>
      </div>

      {/* Scene column headers */}
      <div className="lyrics-editor-row lyrics-editor-scene-header">
        <span className="lyrics-editor-line" />
        <div className="lyrics-editor-scenes">
          {sceneOptions.map((i) => {
            const s = scenes[i];
            return (
              <div key={i} className="lyrics-scene-col-header" title={s ? `${formatTime(s.start)}–${formatTime(s.end)}${s.section ? ` (${s.section})` : ""}` : ""}>
                {i + 1}
              </div>
            );
          })}
        </div>
      </div>

      <div className="lyrics-editor-list">
        {assignments.map((a, lineIdx) => {
          const isHeader = a.is_header || HEADER_PATTERN.test(a.line);
          if (isHeader) {
            return (
              <div key={lineIdx} className="lyrics-editor-row lyrics-editor-section-header">
                <span className="lyrics-editor-section-label">{a.line}</span>
              </div>
            );
          }
          return (
            <div key={lineIdx} className="lyrics-editor-row">
              <span className="lyrics-editor-line" title={a.line}>
                {a.line}
              </span>
              <div className="lyrics-editor-scenes">
                {sceneOptions.map((sceneIdx) => {
                  const active = a.scene_indices.includes(sceneIdx);
                  return (
                    <button
                      key={sceneIdx}
                      className={`lyrics-scene-btn ${active ? "active" : ""}`}
                      onClick={() => toggleScene(lineIdx, sceneIdx)}
                      title={`Scene ${sceneIdx + 1}`}
                    >
                      {sceneIdx + 1}
                    </button>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
