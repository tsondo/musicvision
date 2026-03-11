/**
 * LyricsLineEditor — Manual assignment of lyrics lines to scenes.
 *
 * Shows each lyrics line with a multi-select dropdown to assign it to one
 * or more scenes. Lines are parsed from the lyrics file (section markers
 * like "(Verse 1)" are excluded). Assignments are persisted and passed
 * to the backend on scene creation to skip Whisper transcription.
 */

import { useCallback } from "react";

interface LyricsAssignment {
  line: string;
  scene_indices: number[];
}

interface Props {
  assignments: LyricsAssignment[];
  sceneCount: number;
  onChange: (assignments: LyricsAssignment[]) => void;
}

export default function LyricsLineEditor({ assignments, sceneCount, onChange }: Props) {
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
    if (assignments.length === 0 || sceneCount === 0) return;
    // Distribute lines roughly evenly across scenes
    const linesPerScene = Math.max(1, Math.ceil(assignments.length / sceneCount));
    const updated = assignments.map((a, i) => {
      const sceneIdx = Math.min(Math.floor(i / linesPerScene), sceneCount - 1);
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

  // Build scene labels
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
      <div className="lyrics-editor-list">
        {assignments.map((a, lineIdx) => (
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
        ))}
      </div>
    </div>
  );
}
