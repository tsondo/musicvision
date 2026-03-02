import { useCallback, useMemo, useState } from "react";
import {
  uploadAudio as apiUploadAudio,
  uploadLyrics as apiUploadLyrics,
  runIntake as apiRunIntake,
  generateAllImages as apiGenerateImages,
  generateAllVideos as apiGenerateVideos,
  ApiError,
} from "../api/client";
import type { BatchGenResult, PipelineStage, ProjectConfig, Scene } from "../api/types";

export type StepStatus = "idle" | "running" | "done" | "error";

export interface PipelineState {
  stage: PipelineStage;
  uploadStatus: StepStatus;
  intakeStatus: StepStatus;
  imagesStatus: StepStatus;
  videosStatus: StepStatus;
  error: string | null;
  lastResult: BatchGenResult | null;
  hasAudio: boolean;
  hasLyrics: boolean;
  sceneCount: number;
  imagesRemaining: number;
  videosRemaining: number;
  isRunning: boolean;
}

export function usePipeline(
  config: ProjectConfig | null,
  scenes: Scene[],
  reloadConfig: () => Promise<void>,
  reloadScenes: () => Promise<void>,
) {
  const [uploadStatus, setUploadStatus] = useState<StepStatus>("idle");
  const [intakeStatus, setIntakeStatus] = useState<StepStatus>("idle");
  const [imagesStatus, setImagesStatus] = useState<StepStatus>("idle");
  const [videosStatus, setVideosStatus] = useState<StepStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<BatchGenResult | null>(null);

  const hasAudio = Boolean(config?.song.audio_file);
  const hasLyrics = Boolean(config?.song.lyrics_file);
  const sceneCount = scenes.length;
  const imagesRemaining = scenes.filter((s) => !s.reference_image).length;
  const videosRemaining = scenes.filter(
    (s) => !s.video_clip && !s.sub_clips.some((sc) => sc.video_clip),
  ).length;

  const stage = useMemo<PipelineStage>(() => {
    if (!hasAudio) return "upload";
    if (sceneCount === 0) return "intake";
    if (imagesRemaining > 0) return "images";
    return "videos";
  }, [hasAudio, sceneCount, imagesRemaining]);

  const isRunning =
    uploadStatus === "running" ||
    intakeStatus === "running" ||
    imagesStatus === "running" ||
    videosStatus === "running";

  const uploadAudio = useCallback(
    async (file: File) => {
      setUploadStatus("running");
      setError(null);
      try {
        await apiUploadAudio(file);
        await reloadConfig();
        setUploadStatus("done");
      } catch (err) {
        const msg = err instanceof ApiError ? err.detail : String(err);
        setError(msg);
        setUploadStatus("error");
      }
    },
    [reloadConfig],
  );

  const uploadLyrics = useCallback(
    async (file: File) => {
      setUploadStatus("running");
      setError(null);
      try {
        await apiUploadLyrics(file);
        await reloadConfig();
        setUploadStatus("done");
      } catch (err) {
        const msg = err instanceof ApiError ? err.detail : String(err);
        setError(msg);
        setUploadStatus("error");
      }
    },
    [reloadConfig],
  );

  const runIntake = useCallback(
    async (opts?: { useLlm?: boolean; skipTranscription?: boolean }) => {
      setIntakeStatus("running");
      setError(null);
      try {
        await apiRunIntake(opts);
        await reloadScenes();
        setIntakeStatus("done");
      } catch (err) {
        const msg = err instanceof ApiError ? err.detail : String(err);
        setError(msg);
        setIntakeStatus("error");
      }
    },
    [reloadScenes],
  );

  const generateImages = useCallback(
    async (sceneIds?: string[]) => {
      setImagesStatus("running");
      setError(null);
      try {
        const result = await apiGenerateImages(sceneIds);
        setLastResult(result);
        await reloadScenes();
        setImagesStatus(result.failed.length > 0 ? "error" : "done");
        if (result.failed.length > 0) {
          setError(
            `${result.failed.length} image(s) failed: ${result.failed[0]?.error}`,
          );
        }
      } catch (err) {
        const msg = err instanceof ApiError ? err.detail : String(err);
        setError(msg);
        setImagesStatus("error");
      }
    },
    [reloadScenes],
  );

  const generateVideos = useCallback(
    async (sceneIds?: string[]) => {
      setVideosStatus("running");
      setError(null);
      try {
        const result = await apiGenerateVideos(sceneIds);
        setLastResult(result);
        await reloadScenes();
        setVideosStatus(result.failed.length > 0 ? "error" : "done");
        if (result.failed.length > 0) {
          setError(
            `${result.failed.length} video(s) failed: ${result.failed[0]?.error}`,
          );
        }
      } catch (err) {
        const msg = err instanceof ApiError ? err.detail : String(err);
        setError(msg);
        setVideosStatus("error");
      }
    },
    [reloadScenes],
  );

  const pipelineState: PipelineState = {
    stage,
    uploadStatus,
    intakeStatus,
    imagesStatus,
    videosStatus,
    error,
    lastResult,
    hasAudio,
    hasLyrics,
    sceneCount,
    imagesRemaining,
    videosRemaining,
    isRunning,
  };

  return {
    ...pipelineState,
    uploadAudio,
    uploadLyrics,
    runIntake,
    generateImages,
    generateVideos,
  };
}
