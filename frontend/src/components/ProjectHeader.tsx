import type { PipelineStage, ProjectConfig, Scene } from "../api/types";

interface Props {
  config: ProjectConfig;
  scenes: Scene[];
  stage: PipelineStage;
  onClose: () => void;
}

function formatDuration(seconds: number | null): string {
  if (seconds == null) return "--:--";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

const STAGE_LABELS: Record<PipelineStage, string> = {
  upload: "Upload",
  analyze: "Analyze",
  scenes: "Scenes",
  images: "Images",
  videos: "Videos",
  upscale: "Upscale",
  assembly: "Assembly",
};

export default function ProjectHeader({ config, scenes, stage, onClose }: Props) {
  const withImages = scenes.filter((s) => s.reference_image).length;
  const withVideo = scenes.filter(
    (s) => s.video_clip || s.sub_clips.some((sc) => sc.video_clip),
  ).length;

  return (
    <header className="project-header">
      <div className="header-info">
        <div className="header-title-row">
          <h1>{config.name}</h1>
          <button className="close-project-btn" onClick={onClose} title="Close project">
            &times;
          </button>
        </div>
        <div className="header-stats">
          {config.song.bpm && <span className="stat">BPM {config.song.bpm}</span>}
          {config.song.keyscale && <span className="stat">{config.song.keyscale}</span>}
          <span className="stat">
            {formatDuration(config.song.duration_seconds)}
          </span>
          <span className="stat">{scenes.length} scenes</span>
          <span className="stat">
            {withImages}/{scenes.length} images
          </span>
          <span className="stat">
            {withVideo}/{scenes.length} video
          </span>
          <span className="stage-badge">{STAGE_LABELS[stage]}</span>
        </div>
      </div>
    </header>
  );
}
