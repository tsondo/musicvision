import { useProject } from "./hooks/useProject";
import { useScenes } from "./hooks/useScenes";
import ProjectOpener from "./components/ProjectOpener";
import ProjectHeader from "./components/ProjectHeader";
import Storyboard from "./components/Storyboard";
import PreviewPanel from "./components/PreviewPanel";

export default function App() {
  const { state, open, lastProjectPath } = useProject();
  const projectLoaded = state.status === "loaded";
  const {
    scenes,
    loading,
    generating,
    updateScene,
    regenerateImage,
    regenerateVideo,
  } = useScenes(projectLoaded);

  if (state.status === "loading" || (projectLoaded && loading)) {
    return <div className="loading">Loading...</div>;
  }

  if (state.status === "no-project" || state.status === "error") {
    return (
      <ProjectOpener
        onOpen={open}
        lastPath={lastProjectPath}
        error={state.status === "error" ? state.message : undefined}
      />
    );
  }

  if (state.status !== "loaded") return null;

  return (
    <div className="app">
      <ProjectHeader config={state.config} scenes={scenes} />
      <main>
        <Storyboard
          scenes={scenes}
          generating={generating}
          onUpdate={updateScene}
          onRegenImage={regenerateImage}
          onRegenVideo={regenerateVideo}
        />
      </main>
      <PreviewPanel />
    </div>
  );
}
