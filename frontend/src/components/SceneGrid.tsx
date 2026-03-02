import type { Scene, UpdateSceneRequest } from "../api/types";
import SceneCard from "./SceneCard";

interface Props {
  scenes: Scene[];
  onUpdate: (sceneId: string, updates: UpdateSceneRequest) => Promise<Scene>;
}

export default function SceneGrid({ scenes, onUpdate }: Props) {
  if (scenes.length === 0) {
    return (
      <div className="empty-state">
        No scenes yet. Run the intake pipeline first.
      </div>
    );
  }

  return (
    <div className="scene-grid">
      {scenes.map((scene) => (
        <SceneCard key={scene.id} scene={scene} onUpdate={onUpdate} />
      ))}
    </div>
  );
}
