import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  uploadAudio as apiUploadAudio,
  uploadLyrics as apiUploadLyrics,
  importAudio as apiImportAudio,
  importLyrics as apiImportLyrics,
  runIntake as apiRunIntake,
  generateAllImages as apiGenerateImages,
  generateAllVideos as apiGenerateVideos,
  upscaleVideos as apiUpscaleVideos,
  assemblePreview as apiAssemble,
  ApiError,
} from "../api/client";
import type { AssembleResult, BatchGenResult, ImageModelType, PipelineStage, ProjectConfig, RenderMode, Scene, TargetResolution, UpscalerType, VideoEngineType } from "../api/types";

export type StepStatus = "idle" | "running" | "done" | "error";

export interface PipelineState {
  stage: PipelineStage;
  uploadStatus: StepStatus;
  intakeStatus: StepStatus;
  imagesStatus: StepStatus;
  videosStatus: StepStatus;
  upscaleStatus: StepStatus;
  assembleStatus: StepStatus;
  assembleResult: AssembleResult | null;
  error: string | null;
  lastResult: BatchGenResult | null;
  hasAudio: boolean;
  hasLyrics: boolean;
  sceneCount: number;
  imagesRemaining: number;
  videosRemaining: number;
  upscaleRemaining: number;
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
  const [upscaleStatus, setUpscaleStatus] = useState<StepStatus>("idle");
  const [assembleStatus, setAssembleStatus] = useState<StepStatus>("idle");
  const [assembleResult, setAssembleResult] = useState<AssembleResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<BatchGenResult | null>(null);

  const hasAudio = Boolean(config?.song.audio_file);
  const hasLyrics = Boolean(config?.song.lyrics_file);
  const sceneCount = scenes.length;
  const imagesRemaining = scenes.filter((s) => !s.reference_image).length;
  const videosRemaining = scenes.filter(
    (s) => !s.video_clip && !s.sub_clips.some((sc) => sc.video_clip),
  ).length;
  const upscaleRemaining = scenes.filter(
    (s) =>
      (s.video_clip || s.sub_clips.some((sc) => sc.video_clip)) &&
      !s.upscaled_clip,
  ).length;

  const hasVideos = scenes.some(
    (s) => s.video_clip || s.sub_clips.some((sc) => sc.video_clip),
  );

  const stage = useMemo<PipelineStage>(() => {
    if (!hasAudio) return "upload";
    if (sceneCount === 0) return "intake";
    if (imagesRemaining > 0 && !hasVideos) return "images";
    if (!hasVideos) return "videos";
    if (videosRemaining > 0) return "videos";
    if (upscaleRemaining > 0) return "upscale";
    return "assembly";
  }, [hasAudio, sceneCount, imagesRemaining, videosRemaining, upscaleRemaining, hasVideos]);

  const isRunning =
    uploadStatus === "running" ||
    intakeStatus === "running" ||
    imagesStatus === "running" ||
    videosStatus === "running" ||
    upscaleStatus === "running" ||
    assembleStatus === "running";

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

  const importAudio = useCallback(
    async (path: string) => {
      setUploadStatus("running");
      setError(null);
      try {
        await apiImportAudio(path);
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

  const importLyrics = useCallback(
    async (path: string) => {
      setUploadStatus("running");
      setError(null);
      try {
        await apiImportLyrics(path);
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
    async (sceneIds?: string[], model?: ImageModelType) => {
      setImagesStatus("running");
      setError(null);
      try {
        const result = await apiGenerateImages(sceneIds, model);
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
    async (sceneIds?: string[], engine?: VideoEngineType, renderMode?: RenderMode) => {
      setVideosStatus("running");
      setError(null);
      try {
        const result = await apiGenerateVideos(sceneIds, engine, renderMode);
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

  const upscaleAll = useCallback(
    async (sceneIds?: string[], resolution?: TargetResolution, upscaler?: UpscalerType, renderMode?: RenderMode) => {
      setUpscaleStatus("running");
      setError(null);
      try {
        const result = await apiUpscaleVideos(sceneIds, resolution, upscaler, renderMode);
        await reloadScenes();
        setUpscaleStatus(result.failed.length > 0 ? "error" : "done");
        if (result.failed.length > 0) {
          setError(`${result.failed.length} upscale(s) failed`);
        }
      } catch (err) {
        const msg = err instanceof ApiError ? err.detail : String(err);
        setError(msg);
        setUpscaleStatus("error");
      }
    },
    [reloadScenes],
  );

  const assemble = useCallback(
    async (approvedOnly?: boolean) => {
      setAssembleStatus("running");
      setError(null);
      try {
        const result = await apiAssemble(approvedOnly);
        setAssembleResult(result);
        setAssembleStatus("done");
      } catch (err) {
        const msg = err instanceof ApiError ? err.detail : String(err);
        setError(msg);
        setAssembleStatus("error");
      }
    },
    [],
  );

  // Poll scenes while generation is running so new images/videos appear incrementally
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  useEffect(() => {
    const generating = imagesStatus === "running" || videosStatus === "running" || upscaleStatus === "running";
    if (generating && !pollRef.current) {
      pollRef.current = setInterval(() => {
        reloadScenes();
      }, 3000);
    }
    if (!generating && pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [imagesStatus, videosStatus, upscaleStatus, reloadScenes]);

  const pipelineState: PipelineState = {
    stage,
    uploadStatus,
    intakeStatus,
    imagesStatus,
    videosStatus,
    upscaleStatus,
    assembleStatus,
    assembleResult,
    error,
    lastResult,
    hasAudio,
    hasLyrics,
    sceneCount,
    imagesRemaining,
    videosRemaining,
    upscaleRemaining,
    isRunning,
  };

  return {
    ...pipelineState,
    uploadAudio,
    uploadLyrics,
    importAudio,
    importLyrics,
    runIntake,
    generateImages,
    generateVideos,
    upscaleAll,
    assemble,
  };
}
