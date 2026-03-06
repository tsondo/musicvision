import type {
  RegenerateImageRequest,
  RegenerateVideoRequest,
  Scene,
  UpdateSceneRequest,
} from "../api/types";
import type { SceneGenStatus } from "../hooks/useScenes";
import SceneRow from "./SceneRow";

interface Props {
  scenes: Scene[];
  imageGenStatus: (sceneId: string) => SceneGenStatus;
  videoGenStatus: (sceneId: string) => SceneGenStatus;
  pipelineRunning?: boolean;
  onUpdate: (sceneId: string, updates: UpdateSceneRequest) => Promise<Scene>;
  onRegenImage: (sceneId: string, req: RegenerateImageRequest) => void;
  onRegenVideo: (sceneId: string, req: RegenerateVideoRequest) => void;
  onDequeueImage: (sceneId: string) => void;
  onDequeueVideo: (sceneId: string) => void;
}

export default function Storyboard({
  scenes,
  imageGenStatus,
  videoGenStatus,
  pipelineRunning,
  onUpdate,
  onRegenImage,
  onRegenVideo,
  onDequeueImage,
  onDequeueVideo,
}: Props) {
  if (scenes.length === 0) {
    return (
      <div className="empty-state">
        No scenes yet. Use the pipeline controls above to run intake.
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
          imageGenStatus={imageGenStatus(scene.id)}
          videoGenStatus={videoGenStatus(scene.id)}
          disabled={pipelineRunning}
          onUpdate={onUpdate}
          onRegenImage={onRegenImage}
          onRegenVideo={onRegenVideo}
          onDequeueImage={onDequeueImage}
          onDequeueVideo={onDequeueVideo}
        />
      ))}
    </div>
  );
}
