import { useRef, useState } from "react";
import type { StepStatus } from "../hooks/usePipeline";
import type { AssembleResult, ImageModelType, PipelineStage, RenderMode, TargetResolution, VideoEngineType } from "../api/types";
import FileBrowser from "./FileBrowser";

function progressStyle(done: number, total: number): React.CSSProperties {
  if (total === 0) return {};
  const pct = Math.round((done / total) * 100);
  return {
    background: `linear-gradient(to right, var(--accent) ${pct}%, var(--bg-input) ${pct}%)`,
  };
}

interface Props {
  stage: PipelineStage;
  hasAudio: boolean;
  hasLyrics: boolean;
  analyzed: boolean;
  sceneCount: number;
  imagesRemaining: number;
  videosRemaining: number;
  videosUnapproved: number;
  unapprovedSceneIds: string[];
  upscaleRemaining: number;
  uploadStatus: StepStatus;
  analyzeStatus: StepStatus;
  scenesStatus: StepStatus;
  intakeStatus: StepStatus;
  imagesStatus: StepStatus;
  videosStatus: StepStatus;
  upscaleStatus: StepStatus;
  assembleStatus: StepStatus;
  assembleResult: AssembleResult | null;
  error: string | null;
  isRunning: boolean;
  queueActive: boolean;
  queueDone: number;
  queueTotal: number;
  batchDone: number;
  batchTotal: number;
  onUploadAudio: (file: File) => void;
  onUploadLyrics: (file: File) => void;
  onImportAudio: (path: string) => void;
  importMessage: string | null;
  onImportLyrics: (path: string) => void;
  onRunAnalyze: (opts?: { skipTranscription?: boolean }) => void;
  onRunIntake: (opts?: { useLlm?: boolean; skipTranscription?: boolean }) => void;
  onToggleWaveformEditor: () => void;
  showWaveformEditor: boolean;
  onGenerateImages: (sceneIds?: string[], model?: ImageModelType) => void;
  onGenerateVideos: (sceneIds?: string[], engine?: VideoEngineType, renderMode?: RenderMode) => void;
  onUpscaleVideos: (sceneIds?: string[], resolution?: TargetResolution) => void;
  onAssemble: (approvedOnly?: boolean) => void;
}

type StepState = "disabled" | "active" | "done";

function stepState(
  stepStage: PipelineStage,
  currentStage: PipelineStage,
  status: StepStatus,
  upscaleRemaining?: number,
): StepState {
  if (status === "done") return "done";
  if (stepStage === "upscale" && upscaleRemaining && upscaleRemaining > 0) return "active";
  const order: PipelineStage[] = ["upload", "analyze", "scenes", "images", "videos", "upscale", "assembly"];
  const stepIdx = order.indexOf(stepStage);
  const currentIdx = order.indexOf(currentStage);
  if (stepIdx < currentIdx) return "done";
  if (stepIdx === currentIdx) return "active";
  return "disabled";
}

export default function PipelineBar({
  stage,
  hasAudio,
  hasLyrics,
  analyzed,
  sceneCount,
  imagesRemaining,
  videosRemaining,
  videosUnapproved,
  unapprovedSceneIds,
  upscaleRemaining,
  uploadStatus,
  analyzeStatus,
  scenesStatus,
  intakeStatus,
  imagesStatus,
  videosStatus,
  upscaleStatus,
  assembleStatus,
  assembleResult,
  error,
  isRunning,
  queueActive,
  queueDone,
  queueTotal,
  batchDone,
  batchTotal,
  onUploadAudio,
  onUploadLyrics,
  onImportAudio,
  importMessage,
  onImportLyrics,
  onRunAnalyze,
  onRunIntake: _onRunIntake,
  onToggleWaveformEditor,
  showWaveformEditor,
  onGenerateImages,
  onGenerateVideos,
  onUpscaleVideos,
  onAssemble,
}: Props) {
  const audioRef = useRef<HTMLInputElement>(null);
  const lyricsRef = useRef<HTMLInputElement>(null);
  const [browseTarget, setBrowseTarget] = useState<"audio" | "lyrics" | null>(null);
  const [imageModel, setImageModel] = useState<ImageModelType>("z-image-turbo");
  const [videoEngine, setVideoEngine] = useState<VideoEngineType>("humo");
  const [renderMode, setRenderMode] = useState<RenderMode>("preview");
  const [targetResolution, setTargetResolution] = useState<TargetResolution>("1080p");

  const handleAudioChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onUploadAudio(file);
    e.target.value = "";
  };

  const handleLyricsChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onUploadLyrics(file);
    e.target.value = "";
  };

  const imagesWithCount = sceneCount - imagesRemaining;
  const videosWithCount = sceneCount - videosRemaining;

  // Combined analyze status (analyze or legacy intake)
  const analyzeStepStatus = analyzeStatus !== "idle" ? analyzeStatus : intakeStatus;

  return (
    <div className="pipeline-bar">
      {browseTarget && (
        <FileBrowser
          mode="file"
          fileFilter={
            browseTarget === "audio"
              ? [".flac", ".wav", ".mp3", ".ogg", ".m4a", ".aac"]
              : [".txt", ".lrc"]
          }
          title={browseTarget === "audio" ? "Select Audio File" : "Select Lyrics File"}
          onSelect={(path) => {
            if (browseTarget === "audio") onImportAudio(path);
            else onImportLyrics(path);
            setBrowseTarget(null);
          }}
          onCancel={() => setBrowseTarget(null)}
        />
      )}

      {/* Step 1: Import */}
      <div className={`pipeline-step ${stepState("upload", stage, uploadStatus)}`}>
        <div className="step-label">
          <span className="step-num">1</span> Import
          {hasAudio && <span className="step-check" title="Audio imported" />}
        </div>
        <div className="step-controls">
          <input
            ref={audioRef}
            type="file"
            accept="audio/*"
            hidden
            onChange={handleAudioChange}
          />
          <input
            ref={lyricsRef}
            type="file"
            accept=".txt,.lrc"
            hidden
            onChange={handleLyricsChange}
          />
          <button
            className={`btn-sm ${hasAudio ? "btn-secondary" : ""}`}
            disabled={isRunning}
            onClick={() => setBrowseTarget("audio")}
          >
            {uploadStatus === "running"
              ? "Importing..."
              : hasAudio
                ? "Replace Audio"
                : "Audio"}
          </button>
          <button
            className={`btn-sm btn-secondary`}
            disabled={isRunning}
            onClick={() => setBrowseTarget("lyrics")}
          >
            {hasLyrics ? "Replace Lyrics" : "Lyrics"}
          </button>
          {importMessage && (
            <span className="import-message">{importMessage}</span>
          )}
        </div>
      </div>

      <div className="step-arrow" />

      {/* Step 2: Analyze */}
      <div className={`pipeline-step ${stepState("analyze", stage, analyzeStepStatus)}`}>
        <div className="step-label">
          <span className="step-num">2</span> Analyze
          {analyzed && <span className="step-check" title="Analysis complete" />}
        </div>
        <div className="step-controls">
          <button
            className="btn-sm"
            disabled={!hasAudio || isRunning}
            onClick={() => onRunAnalyze()}
          >
            {analyzeStatus === "running"
              ? "Analyzing..."
              : analyzed
                ? "Re-analyze"
                : "Analyze Audio"}
          </button>
        </div>
      </div>

      <div className="step-arrow" />

      {/* Step 3: Scenes */}
      <div className={`pipeline-step ${stepState("scenes", stage, scenesStatus)}`}>
        <div className="step-label">
          <span className="step-num">3</span> Scenes
          {sceneCount > 0 && (
            <span className="step-check" title={`${sceneCount} scenes`}>
              {sceneCount}
            </span>
          )}
        </div>
        <div className="step-controls">
          <button
            className={`btn-sm ${showWaveformEditor ? "btn-active" : ""}`}
            disabled={!analyzed || isRunning}
            onClick={onToggleWaveformEditor}
          >
            {scenesStatus === "running"
              ? "Creating..."
              : showWaveformEditor
                ? "Hide Editor"
                : sceneCount > 0
                  ? "Edit Scenes"
                  : "Divide Scenes"}
          </button>
        </div>
      </div>

      <div className="step-arrow" />

      {/* Step 4: Images */}
      <div className={`pipeline-step ${stepState("images", stage, imagesStatus)}`}>
        <div className="step-label">
          <span className="step-num">4</span> Images
          {imagesWithCount > 0 && (
            <span className="step-check" title={`${imagesWithCount}/${sceneCount} generated`}>
              {imagesWithCount}/{sceneCount}
            </span>
          )}
        </div>
        <div className="step-controls">
          <select
            className="pipeline-select"
            value={imageModel}
            onChange={(e) => setImageModel(e.target.value as ImageModelType)}
            disabled={isRunning}
          >
            <option value="z-image-turbo">Z-Image Turbo</option>
            <option value="z-image">Z-Image</option>
            <option value="flux-dev">FLUX Dev</option>
            <option value="flux-schnell">FLUX Schnell</option>
          </select>
          <button
            className="btn-sm"
            disabled={sceneCount === 0 || isRunning || imagesRemaining === 0}
            onClick={() => onGenerateImages(undefined, imageModel)}
            style={imagesStatus === "running" ? progressStyle(imagesWithCount, sceneCount) : undefined}
          >
            {imagesStatus === "running"
              ? `Generating ${imagesWithCount}/${sceneCount}...`
              : imagesRemaining > 0
                ? `Generate ${imagesRemaining} Image${imagesRemaining > 1 ? "s" : ""}`
                : "All Done"}
          </button>
        </div>
      </div>

      <div className="step-arrow" />

      {/* Step 5: Videos */}
      <div className={`pipeline-step ${stepState("videos", stage, videosStatus)}`}>
        <div className="step-label">
          <span className="step-num">5</span> Videos
          {videosWithCount > 0 && (
            <span className="step-check" title={`${videosWithCount}/${sceneCount} generated`}>
              {videosWithCount}/{sceneCount}
            </span>
          )}
        </div>
        <div className="step-controls">
          <select
            className="pipeline-select"
            value={videoEngine}
            onChange={(e) => setVideoEngine(e.target.value as VideoEngineType)}
            disabled={isRunning}
          >
            <option value="humo">HuMo</option>
            <option value="ltx_video">LTX-Video 2</option>
            <option value="hunyuan_avatar">HunyuanVideo Avatar</option>
          </select>
          <div className="render-mode-toggle">
            <button
              className={`btn-toggle${renderMode === "preview" ? " active" : ""}`}
              onClick={() => setRenderMode("preview")}
              disabled={isRunning}
              title="256p / 10 steps — fast preview"
            >
              Preview
            </button>
            <button
              className={`btn-toggle${renderMode === "final" ? " active" : ""}`}
              onClick={() => setRenderMode("final")}
              disabled={isRunning}
              title="512p / 30 steps — final quality"
            >
              Final
            </button>
          </div>
          <button
            className="btn-sm"
            disabled={
              sceneCount === 0 ||
              isRunning ||
              imagesRemaining > 0 ||
              (videosRemaining === 0 && videosUnapproved === 0)
            }
            onClick={() => {
              if (videosRemaining > 0) {
                onGenerateVideos(undefined, videoEngine, renderMode);
              } else {
                onGenerateVideos(unapprovedSceneIds, videoEngine, renderMode);
              }
            }}
            style={videosStatus === "running"
              ? progressStyle(batchTotal > 0 ? batchDone : videosWithCount, batchTotal > 0 ? batchTotal : sceneCount)
              : undefined}
          >
            {videosStatus === "running"
              ? `Rendering ${batchTotal > 0 ? batchDone : videosWithCount}/${batchTotal > 0 ? batchTotal : sceneCount}...`
              : videosRemaining > 0
                ? `Render ${videosRemaining} ${renderMode} video${videosRemaining > 1 ? "s" : ""}`
                : videosUnapproved > 0
                  ? `Re-render ${videosUnapproved} unapproved`
                  : "All Approved"}
          </button>
        </div>
      </div>

      <div className="step-arrow" />

      {/* Step 6: Upscale */}
      <div className={`pipeline-step ${stepState("upscale", stage, upscaleStatus, upscaleRemaining)}`}>
        <div className="step-label">
          <span className="step-num">6</span> Upscale
          {sceneCount - upscaleRemaining > 0 && (
            <span className="step-check" title={`${sceneCount - upscaleRemaining}/${sceneCount} upscaled`}>
              {sceneCount - upscaleRemaining}/{sceneCount}
            </span>
          )}
        </div>
        <div className="step-controls">
          <select
            className="pipeline-select"
            value={targetResolution}
            onChange={(e) => setTargetResolution(e.target.value as TargetResolution)}
            disabled={isRunning}
          >
            <option value="720p">720p</option>
            <option value="1080p">1080p</option>
            <option value="1440p" disabled>1440p (48GB+ VRAM)</option>
            <option value="4k" disabled>4K (48GB+ VRAM)</option>
          </select>
          <button
            className="btn-sm"
            disabled={sceneCount === 0 || isRunning || upscaleRemaining === 0}
            onClick={() => onUpscaleVideos(undefined, targetResolution)}
            style={upscaleStatus === "running" ? progressStyle(sceneCount - upscaleRemaining, sceneCount) : undefined}
          >
            {upscaleStatus === "running"
              ? `Upscaling ${sceneCount - upscaleRemaining}/${sceneCount}...`
              : upscaleRemaining > 0
                ? `Upscale ${upscaleRemaining} clip${upscaleRemaining > 1 ? "s" : ""}`
                : "All Done"}
          </button>
        </div>
      </div>

      <div className="step-arrow" />

      {/* Step 7: Assembly */}
      <div className={`pipeline-step ${stepState("assembly", stage, assembleStatus)}`}>
        <div className="step-label">
          <span className="step-num">7</span> Assembly
          {assembleResult && <span className="step-check" title="Assembled" />}
        </div>
        <div className="step-controls">
          <button
            className="btn-sm"
            disabled={sceneCount === 0 || videosRemaining === sceneCount || isRunning}
            onClick={() => onAssemble()}
          >
            {assembleStatus === "running"
              ? "Assembling..."
              : assembleResult
                ? "Reassemble"
                : "Assemble"}
          </button>
        </div>
      </div>

      {queueActive && queueTotal > 0 && (
        <div className="queue-progress">
          <button
            className="btn-sm btn-queue-progress"
            disabled
            style={progressStyle(queueDone, queueTotal)}
          >
            Queue: {queueDone}/{queueTotal} complete
          </button>
        </div>
      )}

      {error && <div className="pipeline-error">{error}</div>}
    </div>
  );
}
