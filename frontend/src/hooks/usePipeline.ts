import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  uploadAudio as apiUploadAudio,
  uploadLyrics as apiUploadLyrics,
  importAudio as apiImportAudio,
  importLyrics as apiImportLyrics,
  runIntake as apiRunIntake,
  runAnalyze as apiRunAnalyze,
  getAnalysis as apiGetAnalysis,
  createScenes as apiCreateScenes,
  autoSegment as apiAutoSegment,
  generateAllImages as apiGenerateImages,
  generateAllVideos as apiGenerateVideos,
  upscaleVideos as apiUpscaleVideos,
  assemblePreview as apiAssemble,
  ApiError,
} from "../api/client";
import type { AnalysisResult, AssembleResult, BatchGenResult, ImageModelType, PipelineStage, ProjectConfig, RenderMode, Scene, SceneBoundary, TargetResolution, UpscalerType, VideoEngineType } from "../api/types";

export type StepStatus = "idle" | "running" | "done" | "error";

export interface PipelineState {
  stage: PipelineStage;
  uploadStatus: StepStatus;
  analyzeStatus: StepStatus;
  scenesStatus: StepStatus;
  intakeStatus: StepStatus;
  imagesStatus: StepStatus;
  videosStatus: StepStatus;
  upscaleStatus: StepStatus;
  assembleStatus: StepStatus;
  assembleResult: AssembleResult | null;
  analysisResult: AnalysisResult | null;
  error: string | null;
  lastResult: BatchGenResult | null;
  hasAudio: boolean;
  hasLyrics: boolean;
  analyzed: boolean;
  sceneCount: number;
  imagesRemaining: number;
  videosRemaining: number;
  videosUnapproved: number;
  unapprovedSceneIds: string[];
  upscaleRemaining: number;
  isRunning: boolean;
  batchDone: number;
  batchTotal: number;
}

export function usePipeline(
  config: ProjectConfig | null,
  scenes: Scene[],
  reloadConfig: () => Promise<void>,
  reloadScenes: () => Promise<void>,
) {
  const [uploadStatus, setUploadStatus] = useState<StepStatus>("idle");
  const [analyzeStatus, setAnalyzeStatus] = useState<StepStatus>("idle");
  const [scenesStatus, setScenesStatus] = useState<StepStatus>("idle");
  const [intakeStatus, setIntakeStatus] = useState<StepStatus>("idle");
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [imagesStatus, setImagesStatus] = useState<StepStatus>("idle");
  const [videosStatus, setVideosStatus] = useState<StepStatus>("idle");
  const [upscaleStatus, setUpscaleStatus] = useState<StepStatus>("idle");
  const [assembleStatus, setAssembleStatus] = useState<StepStatus>("idle");
  const [assembleResult, setAssembleResult] = useState<AssembleResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<BatchGenResult | null>(null);

  // Batch progress tracking: snapshot clip paths at batch start, count changes via polling
  const [batchSnapshot, setBatchSnapshot] = useState<{ step: string; clips: Record<string, string | null>; total: number } | null>(null);

  const batchDone = (() => {
    if (!batchSnapshot) return 0;
    let done = 0;
    for (const scene of scenes) {
      const oldClip = batchSnapshot.clips[scene.id];
      const newClip = scene.video_clip || scene.sub_clips.find((sc) => sc.video_clip)?.video_clip || null;
      if (newClip !== oldClip) done++;
    }
    return done;
  })();
  const batchTotal = batchSnapshot?.total ?? 0;

  const hasAudio = Boolean(config?.song.audio_file);
  const hasLyrics = Boolean(config?.song.lyrics_file);
  const analyzed = Boolean(config?.song.analyzed) || Boolean(analysisResult?.analyzed);
  const sceneCount = scenes.length;
  const imagesRemaining = scenes.filter((s) => !s.reference_image).length;
  const videosRemaining = scenes.filter(
    (s) => !s.video_clip && !s.sub_clips.some((sc) => sc.video_clip),
  ).length;
  const unapprovedScenes = scenes.filter(
    (s) =>
      s.video_status !== "approved" &&
      (s.video_clip || s.sub_clips.some((sc) => sc.video_clip)),
  );
  const videosUnapproved = unapprovedScenes.length;
  const unapprovedSceneIds = unapprovedScenes.map((s) => s.id);
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
    if (!analyzed) return "analyze";
    if (sceneCount === 0) return "scenes";
    if (imagesRemaining > 0 && !hasVideos) return "images";
    if (!hasVideos) return "videos";
    if (videosRemaining > 0) return "videos";
    if (upscaleRemaining > 0) return "upscale";
    return "assembly";
  }, [hasAudio, analyzed, sceneCount, imagesRemaining, videosRemaining, upscaleRemaining, hasVideos]);

  const isRunning =
    uploadStatus === "running" ||
    analyzeStatus === "running" ||
    scenesStatus === "running" ||
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

  // Phase 1: Analyze audio (BPM, Whisper, demucs)
  const runAnalyze = useCallback(
    async (opts?: { skipTranscription?: boolean }) => {
      setAnalyzeStatus("running");
      setError(null);
      try {
        const result = await apiRunAnalyze(opts);
        setAnalysisResult(result);
        await reloadConfig();
        setAnalyzeStatus("done");
      } catch (err) {
        const msg = err instanceof ApiError ? err.detail : String(err);
        setError(msg);
        setAnalyzeStatus("error");
      }
    },
    [reloadConfig],
  );

  // Load existing analysis data (on project open when already analyzed)
  const loadAnalysis = useCallback(async () => {
    try {
      const result = await apiGetAnalysis();
      if (result.analyzed) {
        setAnalysisResult(result);
        setAnalyzeStatus("done");
      }
    } catch {
      // Not analyzed yet — that's fine
    }
  }, []);

  // Load analysis on mount if already analyzed
  useEffect(() => {
    if (config?.song.analyzed && !analysisResult) {
      loadAnalysis();
    }
  }, [config?.song.analyzed, analysisResult, loadAnalysis]);

  // Phase 2: Create scenes from manual boundaries
  const confirmScenes = useCallback(
    async (boundaries: SceneBoundary[], snapToBeats: boolean) => {
      setScenesStatus("running");
      setError(null);
      try {
        await apiCreateScenes(boundaries, snapToBeats);
        await reloadScenes();
        setScenesStatus("done");
      } catch (err) {
        const msg = err instanceof ApiError ? err.detail : String(err);
        setError(msg);
        setScenesStatus("error");
      }
    },
    [reloadScenes],
  );

  // Phase 2 alt: Auto-segment
  const runAutoSegment = useCallback(
    async (useLlm: boolean) => {
      setScenesStatus("running");
      setError(null);
      try {
        await apiAutoSegment(useLlm);
        await reloadScenes();
        setScenesStatus("done");
      } catch (err) {
        const msg = err instanceof ApiError ? err.detail : String(err);
        setError(msg);
        setScenesStatus("error");
      }
    },
    [reloadScenes],
  );

  // Legacy one-shot intake (kept for backward compat)
  const runIntake = useCallback(
    async (opts?: { useLlm?: boolean; skipTranscription?: boolean }) => {
      setIntakeStatus("running");
      setError(null);
      try {
        await apiRunIntake(opts);
        await reloadScenes();
        await reloadConfig();
        setIntakeStatus("done");
      } catch (err) {
        const msg = err instanceof ApiError ? err.detail : String(err);
        setError(msg);
        setIntakeStatus("error");
      }
    },
    [reloadScenes, reloadConfig],
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
      // Snapshot current clips to track re-render progress
      const targetScenes = sceneIds ? scenes.filter((s) => sceneIds.includes(s.id)) : scenes;
      const clips: Record<string, string | null> = {};
      for (const s of targetScenes) {
        clips[s.id] = s.video_clip || s.sub_clips.find((sc) => sc.video_clip)?.video_clip || null;
      }
      setBatchSnapshot({ step: "videos", clips, total: targetScenes.length });
      setVideosStatus("running");
      setError(null);
      try {
        const result = await apiGenerateVideos(sceneIds, engine, renderMode);
        setLastResult(result);
        await reloadScenes();
        setBatchSnapshot(null);
        setVideosStatus(result.failed.length > 0 ? "error" : "done");
        if (result.failed.length > 0) {
          setError(
            `${result.failed.length} video(s) failed: ${result.failed[0]?.error}`,
          );
        }
      } catch (err) {
        const msg = err instanceof ApiError ? err.detail : String(err);
        setError(msg);
        setBatchSnapshot(null);
        setVideosStatus("error");
      }
    },
    [scenes, reloadScenes],
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
      }, 10000);
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
    analyzeStatus,
    scenesStatus,
    intakeStatus,
    imagesStatus,
    videosStatus,
    upscaleStatus,
    assembleStatus,
    assembleResult,
    analysisResult,
    error,
    lastResult,
    hasAudio,
    hasLyrics,
    analyzed,
    sceneCount,
    imagesRemaining,
    videosRemaining,
    videosUnapproved,
    unapprovedSceneIds,
    upscaleRemaining,
    isRunning,
    batchDone,
    batchTotal,
  };

  return {
    ...pipelineState,
    uploadAudio,
    uploadLyrics,
    importAudio,
    importLyrics,
    runAnalyze,
    confirmScenes,
    runAutoSegment,
    runIntake,
    generateImages,
    generateVideos,
    upscaleAll,
    assemble,
  };
}
