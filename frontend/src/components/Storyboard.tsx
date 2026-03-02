import type {
  RegenerateImageRequest,
  RegenerateVideoRequest,
  Scene,
  UpdateSceneRequest,
} from "../api/types";
import SceneRow from "./SceneRow";

interface Props {
  scenes: Scene[];
  generating: Set<string>;
  onUpdate: (sceneId: string, updates: UpdateSceneRequest) => Promise<Scene>;
  onRegenImage: (
    sceneId: string,
    req: RegenerateImageRequest,
  ) => Promise<Scene>;
  onRegenVideo: (
    sceneId: string,
    req: RegenerateVideoRequest,
  ) => Promise<Scene>;
}

export default function Storyboard({
  scenes,
  generating,
  onUpdate,
  onRegenImage,
  onRegenVideo,
}: Props) {
  if (scenes.length === 0) {
    return (
      <div className="empty-state">
        No scenes yet. Run the intake pipeline first.
      </div>
    );
  }

  return (
    <div className="storyboard">
      {/* Column headers */}
      <div className="storyboard-header scene-row">
        <div className="cell cell-info">#</div>
        <div className="cell cell-image">Image</div>
        <div className="cell cell-prompt">Image Description</div>
        <div className="cell cell-source">Source</div>
        <div className="cell cell-controls">Generate Image</div>
        <div className="cell cell-prompt">Motion Description</div>
        <div className="cell cell-controls">Generate Video</div>
      </div>

      {scenes.map((scene) => (
        <SceneRow
          key={scene.id}
          scene={scene}
          generating={generating}
          onUpdate={onUpdate}
          onRegenImage={onRegenImage}
          onRegenVideo={onRegenVideo}
        />
      ))}
    </div>
  );
}
