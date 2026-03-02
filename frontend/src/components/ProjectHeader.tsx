import type { ProjectConfig, Scene } from "../api/types";

interface Props {
  config: ProjectConfig;
  scenes: Scene[];
}

function formatDuration(seconds: number | null): string {
  if (seconds == null) return "--:--";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function ProjectHeader({ config, scenes }: Props) {
  const withImages = scenes.filter((s) => s.reference_image).length;
  const withVideo = scenes.filter(
    (s) => s.video_clip || s.sub_clips.some((sc) => sc.video_clip),
  ).length;

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
          <span className="stat">{scenes.length} scenes</span>
          <span className="stat">
            {withImages}/{scenes.length} images
          </span>
          <span className="stat">
            {withVideo}/{scenes.length} video
          </span>
        </div>
      </div>
    </header>
  );
}
