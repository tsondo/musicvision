import { useRef, useState } from "react";
import type { StepStatus } from "../hooks/usePipeline";
import type { PipelineStage } from "../api/types";

interface Props {
  stage: PipelineStage;
  hasAudio: boolean;
  hasLyrics: boolean;
  sceneCount: number;
  imagesRemaining: number;
  videosRemaining: number;
  uploadStatus: StepStatus;
  intakeStatus: StepStatus;
  imagesStatus: StepStatus;
  videosStatus: StepStatus;
  error: string | null;
  isRunning: boolean;
  onUploadAudio: (file: File) => void;
  onUploadLyrics: (file: File) => void;
  onRunIntake: (opts?: { useLlm?: boolean; skipTranscription?: boolean }) => void;
  onGenerateImages: () => void;
  onGenerateVideos: () => void;
}

type StepState = "disabled" | "active" | "done";

function stepState(
  stepStage: PipelineStage,
  currentStage: PipelineStage,
  status: StepStatus,
): StepState {
  const order: PipelineStage[] = ["upload", "intake", "images", "videos"];
  const stepIdx = order.indexOf(stepStage);
  const currentIdx = order.indexOf(currentStage);
  if (status === "done" || stepIdx < currentIdx) return "done";
  if (stepIdx === currentIdx) return "active";
  return "disabled";
}

export default function PipelineBar({
  stage,
  hasAudio,
  hasLyrics,
  sceneCount,
  imagesRemaining,
  videosRemaining,
  uploadStatus,
  intakeStatus,
  imagesStatus,
  videosStatus,
  error,
  isRunning,
  onUploadAudio,
  onUploadLyrics,
  onRunIntake,
  onGenerateImages,
  onGenerateVideos,
}: Props) {
  const audioRef = useRef<HTMLInputElement>(null);
  const lyricsRef = useRef<HTMLInputElement>(null);
  const [useLlm, setUseLlm] = useState(true);

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

  return (
    <div className="pipeline-bar">
      {/* Step 1: Upload */}
      <div className={`pipeline-step ${stepState("upload", stage, uploadStatus)}`}>
        <div className="step-label">
          <span className="step-num">1</span> Upload
          {hasAudio && <span className="step-check" title="Audio uploaded" />}
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
            onClick={() => audioRef.current?.click()}
          >
            {uploadStatus === "running"
              ? "Uploading..."
              : hasAudio
                ? "Replace Audio"
                : "Audio"}
          </button>
          <button
            className={`btn-sm btn-secondary`}
            disabled={isRunning}
            onClick={() => lyricsRef.current?.click()}
          >
            {hasLyrics ? "Replace Lyrics" : "Lyrics"}
          </button>
        </div>
      </div>

      <div className="step-arrow" />

      {/* Step 2: Intake */}
      <div className={`pipeline-step ${stepState("intake", stage, intakeStatus)}`}>
        <div className="step-label">
          <span className="step-num">2</span> Intake
          {sceneCount > 0 && (
            <span className="step-check" title={`${sceneCount} scenes`}>
              {sceneCount}
            </span>
          )}
        </div>
        <div className="step-controls">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={useLlm}
              onChange={(e) => setUseLlm(e.target.checked)}
            />
            LLM segmentation
          </label>
          <button
            className="btn-sm"
            disabled={!hasAudio || isRunning}
            onClick={() => onRunIntake({ useLlm })}
          >
            {intakeStatus === "running" ? "Running..." : "Run Intake"}
          </button>
        </div>
      </div>

      <div className="step-arrow" />

      {/* Step 3: Images */}
      <div className={`pipeline-step ${stepState("images", stage, imagesStatus)}`}>
        <div className="step-label">
          <span className="step-num">3</span> Images
          {imagesWithCount > 0 && (
            <span className="step-check" title={`${imagesWithCount}/${sceneCount} generated`}>
              {imagesWithCount}/{sceneCount}
            </span>
          )}
        </div>
        <div className="step-controls">
          <button
            className="btn-sm"
            disabled={sceneCount === 0 || isRunning || imagesRemaining === 0}
            onClick={() => onGenerateImages()}
          >
            {imagesStatus === "running"
              ? "Generating..."
              : imagesRemaining > 0
                ? `Generate ${imagesRemaining} Image${imagesRemaining > 1 ? "s" : ""}`
                : "All Done"}
          </button>
        </div>
      </div>

      <div className="step-arrow" />

      {/* Step 4: Videos */}
      <div className={`pipeline-step ${stepState("videos", stage, videosStatus)}`}>
        <div className="step-label">
          <span className="step-num">4</span> Videos
          {videosWithCount > 0 && (
            <span className="step-check" title={`${videosWithCount}/${sceneCount} generated`}>
              {videosWithCount}/{sceneCount}
            </span>
          )}
        </div>
        <div className="step-controls">
          <button
            className="btn-sm"
            disabled={sceneCount === 0 || isRunning || imagesRemaining > 0 || videosRemaining === 0}
            onClick={() => onGenerateVideos()}
          >
            {videosStatus === "running"
              ? "Generating..."
              : videosRemaining > 0
                ? `Generate ${videosRemaining} Video${videosRemaining > 1 ? "s" : ""}`
                : "All Done"}
          </button>
        </div>
      </div>

      {error && <div className="pipeline-error">{error}</div>}
    </div>
  );
}
