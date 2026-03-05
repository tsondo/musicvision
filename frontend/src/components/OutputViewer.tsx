import { useState } from "react";
import { fileUrl } from "../api/client";
import type { AssembleResult } from "../api/types";

interface Props {
  result: AssembleResult;
}

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function OutputViewer({ result }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [cacheTs] = useState(() => Date.now());

  const absDrift = Math.abs(result.drift_seconds);
  const driftClass =
    absDrift < 0.1 ? "drift-ok" : absDrift < 1 ? "drift-warn" : "drift-bad";

  return (
    <div className={`output-viewer ${expanded ? "expanded" : ""}`}>
      <div className="output-viewer-bar" onClick={() => setExpanded(!expanded)}>
        <span className="ov-stat">{formatDuration(result.video_duration_seconds)}</span>
        <span className="ov-stat">{result.clip_count} clips</span>
        <span className={`ov-drift ${driftClass}`}>
          drift {result.drift_seconds > 0 ? "+" : ""}
          {result.drift_seconds.toFixed(3)}s
        </span>
        <span className="ov-toggle">{expanded ? "\u25BC" : "\u25B2"}</span>
      </div>
      {expanded && (
        <div className="output-viewer-expanded">
          <video
            className="output-video"
            controls
            src={fileUrl(result.rough_cut, cacheTs)}
          />
          <div className="output-info">
            <div className="output-files">
              <span className="output-files-label">Output files:</span>
              <code>{result.output_dir}/</code>
            </div>
            <div className="output-files-list">
              <span>rough_cut.mp4</span>
              {result.edl && <span>timeline.edl</span>}
              {result.fcpxml && <span>timeline.fcpxml</span>}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
