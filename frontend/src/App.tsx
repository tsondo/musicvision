import { useState } from "react";
import { useProject } from "./hooks/useProject";
import { useScenes } from "./hooks/useScenes";
import { usePipeline } from "./hooks/usePipeline";
import { fileUrl } from "./api/client";
import ProjectOpener from "./components/ProjectOpener";
import ProjectHeader from "./components/ProjectHeader";
import PipelineBar from "./components/PipelineBar";
import Storyboard from "./components/Storyboard";
import OutputViewer from "./components/OutputViewer";
import WaveformEditor from "./components/WaveformEditor";

export default function App() {
  const { state, open, create, close, reload: reloadConfig, lastProjectPath } = useProject();
  const projectLoaded = state.status === "loaded";
  const {
    scenes,
    loading,
    queueActive,
    queueDone,
    queueTotal,
    imageGenStatus,
    videoGenStatus,
    reload: reloadScenes,
    updateScene,
    regenerateImage,
    regenerateVideo,
    dequeueImage,
    dequeueVideo,
  } = useScenes(projectLoaded);

  const pipeline = usePipeline(
    projectLoaded ? state.config : null,
    scenes,
    reloadConfig,
    reloadScenes,
  );

  const [showWaveformEditor, setShowWaveformEditor] = useState(false);

  // Auto-show waveform editor when we're at the "scenes" stage and no scenes exist
  const waveformVisible =
    showWaveformEditor ||
    (pipeline.stage === "scenes" && pipeline.analyzed && pipeline.sceneCount === 0);

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
      <ProjectHeader config={state.config} scenes={scenes} stage={pipeline.stage} onClose={close} />
      <PipelineBar
        stage={pipeline.stage}
        hasAudio={pipeline.hasAudio}
        hasLyrics={pipeline.hasLyrics}
        analyzed={pipeline.analyzed}
        sceneCount={pipeline.sceneCount}
        imagesRemaining={pipeline.imagesRemaining}
        videosRemaining={pipeline.videosRemaining}
        videosUnapproved={pipeline.videosUnapproved}
        unapprovedSceneIds={pipeline.unapprovedSceneIds}
        upscaleRemaining={pipeline.upscaleRemaining}
        uploadStatus={pipeline.uploadStatus}
        analyzeStatus={pipeline.analyzeStatus}
        scenesStatus={pipeline.scenesStatus}
        intakeStatus={pipeline.intakeStatus}
        imagesStatus={pipeline.imagesStatus}
        videosStatus={pipeline.videosStatus}
        upscaleStatus={pipeline.upscaleStatus}
        error={pipeline.error}
        isRunning={pipeline.isRunning}
        queueActive={queueActive}
        queueDone={queueDone}
        queueTotal={queueTotal}
        batchDone={pipeline.batchDone}
        batchTotal={pipeline.batchTotal}
        onUploadAudio={pipeline.uploadAudio}
        onUploadLyrics={pipeline.uploadLyrics}
        onImportAudio={pipeline.importAudio}
        onImportLyrics={pipeline.importLyrics}
        onRunAnalyze={pipeline.runAnalyze}
        onRunIntake={pipeline.runIntake}
        onToggleWaveformEditor={() => setShowWaveformEditor((v) => !v)}
        showWaveformEditor={waveformVisible}
        onGenerateImages={pipeline.generateImages}
        onGenerateVideos={pipeline.generateVideos}
        onUpscaleVideos={pipeline.upscaleAll}
        assembleStatus={pipeline.assembleStatus}
        assembleResult={pipeline.assembleResult}
        onAssemble={pipeline.assemble}
      />

      {/* Waveform editor — shown between pipeline bar and storyboard */}
      {waveformVisible && pipeline.analysisResult && state.config.song.audio_file && (
        <WaveformEditor
          audioUrl={state.config.song.audio_file}
          analysis={pipeline.analysisResult}
          onConfirm={(boundaries, snapToBeats) => {
            pipeline.confirmScenes(boundaries, snapToBeats);
            setShowWaveformEditor(false);
          }}
          onAutoSegment={(useLlm) => {
            pipeline.runAutoSegment(useLlm);
            setShowWaveformEditor(false);
          }}
          isRunning={pipeline.isRunning}
        />
      )}

      <main>
        <Storyboard
          scenes={scenes}
          imageGenStatus={imageGenStatus}
          videoGenStatus={videoGenStatus}
          pipelineRunning={pipeline.isRunning}
          onUpdate={updateScene}
          onRegenImage={regenerateImage}
          onRegenVideo={regenerateVideo}
          onDequeueImage={dequeueImage}
          onDequeueVideo={dequeueVideo}
        />
      </main>
      {pipeline.assembleResult && <OutputViewer result={pipeline.assembleResult} />}
    </div>
  );
}
