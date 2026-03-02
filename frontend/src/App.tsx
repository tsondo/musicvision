import { useProject } from "./hooks/useProject";
import { useScenes } from "./hooks/useScenes";
import { usePipeline } from "./hooks/usePipeline";
import ProjectOpener from "./components/ProjectOpener";
import ProjectHeader from "./components/ProjectHeader";
import PipelineBar from "./components/PipelineBar";
import Storyboard from "./components/Storyboard";
import PreviewPanel from "./components/PreviewPanel";

export default function App() {
  const { state, open, create, reload: reloadConfig, lastProjectPath } = useProject();
  const projectLoaded = state.status === "loaded";
  const {
    scenes,
    loading,
    generating,
    reload: reloadScenes,
    updateScene,
    regenerateImage,
    regenerateVideo,
  } = useScenes(projectLoaded);

  const pipeline = usePipeline(
    projectLoaded ? state.config : null,
    scenes,
    reloadConfig,
    reloadScenes,
  );

  if (state.status === "loading" || (projectLoaded && loading)) {
    return <div className="loading">Loading...</div>;
  }

  if (state.status === "no-project" || state.status === "error") {
    return (
      <ProjectOpener
        onOpen={open}
        onCreate={create}
        lastPath={lastProjectPath}
        error={state.status === "error" ? state.message : undefined}
      />
    );
  }

  if (state.status !== "loaded") return null;

  return (
    <div className="app">
      <ProjectHeader config={state.config} scenes={scenes} stage={pipeline.stage} />
      <PipelineBar
        stage={pipeline.stage}
        hasAudio={pipeline.hasAudio}
        hasLyrics={pipeline.hasLyrics}
        sceneCount={pipeline.sceneCount}
        imagesRemaining={pipeline.imagesRemaining}
        videosRemaining={pipeline.videosRemaining}
        uploadStatus={pipeline.uploadStatus}
        intakeStatus={pipeline.intakeStatus}
        imagesStatus={pipeline.imagesStatus}
        videosStatus={pipeline.videosStatus}
        error={pipeline.error}
        isRunning={pipeline.isRunning}
        onUploadAudio={pipeline.uploadAudio}
        onUploadLyrics={pipeline.uploadLyrics}
        onImportAudio={pipeline.importAudio}
        onImportLyrics={pipeline.importLyrics}
        onRunIntake={pipeline.runIntake}
        onGenerateImages={pipeline.generateImages}
        onGenerateVideos={pipeline.generateVideos}
      />
      <main>
        <Storyboard
          scenes={scenes}
          generating={generating}
          pipelineRunning={pipeline.isRunning}
          onUpdate={updateScene}
          onRegenImage={regenerateImage}
          onRegenVideo={regenerateVideo}
        />
      </main>
      <PreviewPanel />
    </div>
  );
}
