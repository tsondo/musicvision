import type { ProjectConfig, Scene } from "../api/types";

interface Props {
  config: ProjectConfig;
  scenes: Scene[];
  onApproveAll: () => void;
}

function formatDuration(seconds: number | null): string {
  if (seconds == null) return "--:--";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function ProjectHeader({ config, scenes, onApproveAll }: Props) {
  const approvedCount = scenes.filter(
    (s) => s.image_status === "approved",
  ).length;
  const withImages = scenes.filter((s) => s.reference_image).length;
  const subClipCount = scenes.reduce((acc, s) => acc + s.sub_clips.length, 0);

  return (
    <header className="project-header">
      <div className="header-info">
        <h1>{config.name}</h1>
        <div className="header-stats">
          {config.song.bpm && <span className="stat">BPM {config.song.bpm}</span>}
          {config.song.keyscale && <span className="stat">{config.song.keyscale}</span>}
          <span className="stat">
            {formatDuration(config.song.duration_seconds)}
          </span>
          <span className="stat">
            {scenes.length} scenes
            {subClipCount > 0 && ` (${subClipCount} sub-clips)`}
          </span>
          <span className="stat">
            {withImages}/{scenes.length} images
          </span>
          <span className="stat">
            {approvedCount}/{scenes.length} approved
          </span>
        </div>
      </div>
      <div className="header-actions">
        <button onClick={onApproveAll} className="btn-secondary">
          Approve All
        </button>
      </div>
    </header>
  );
}
