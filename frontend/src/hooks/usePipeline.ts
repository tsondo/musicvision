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
  generateDescriptions as apiGenerateDescriptions,
  generateVideoDescriptions as apiGenerateVideoDescriptions,
  generateAllImages as apiGenerateImages,
  generateAllVideos as apiGenerateVideos,
  upscaleVideos as apiUpscaleVideos,
  assemblePreview as apiAssemble,
  cancelJob as apiCancelJob,
  listJobs as apiListJobs,
  ApiError,
  JobCancelledError,
  type JobCallbacks,
} from "../api/client";
import type { AnalysisResult, AssembleResult, BatchGenResult, Job, LyricsSource, PipelineStage, ProjectConfig, RenderMode, Scene, SceneBoundary, TargetResolution, UpscalerType } from "../api/types";

export type StepStatus = "idle" | "running" | "done" | "error";

export interface PipelineState {
  stage: PipelineStage;
  uploadStatus: StepStatus;
  analyzeStatus: StepStatus;
  scenesStatus: StepStatus;
  intakeStatus: StepStatus;
  descriptionsStatus: StepStatus;
  descriptionsRemaining: number;
  imagesStatus: StepStatus;
  videoDescriptionsStatus: StepStatus;
  videoDescriptionsRemaining: number;
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
  /** Scenes that actually have an upscaled clip. */
  upscaledCount: number;
  isRunning: boolean;
  /** True only while scene segmentation itself runs (not any other job). */
  isSegmenting: boolean;
  batchDone: number;
  batchTotal: number;
  /** The queued/running job (ours or one an agent submitted), if any. */
  activeJob: Job | null;
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
  const [descriptionsStatus, setDescriptionsStatus] = useState<StepStatus>("idle");
  const [imagesStatus, setImagesStatus] = useState<StepStatus>("idle");
  const [videoDescriptionsStatus, setVideoDescriptionsStatus] = useState<StepStatus>("idle");
  const [videosStatus, setVideosStatus] = useState<StepStatus>("idle");
  const [upscaleStatus, setUpscaleStatus] = useState<StepStatus>("idle");
  const [assembleStatus, setAssembleStatus] = useState<StepStatus>("idle");
  const [assembleResult, setAssembleResult] = useState<AssembleResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<BatchGenResult | null>(null);

  // Job progress (AGENT_INTERFACE_SPEC.md §4.2): long-running steps are jobs;
  // progress comes from job.progress instead of diffing scene clip paths.
  const [activeJob, setActiveJob] = useState<Job | null>(null);
  const localJobRef = useRef(false);
  const lastProgressRef = useRef("");

  const batchDone = activeJob?.progress.current ?? 0;
  const batchTotal = activeJob?.progress.total ?? 0;

  const jobCallbacks = useMemo<JobCallbacks>(
    () => ({
      onUpdate: (job: Job) => {
        const live = job.state === "queued" || job.state === "running";
        setActiveJob(live ? job : null);
        // Each finished unit (scene) has been saved server-side: refresh so
        // new images/clips appear as they land.
        const key = `${job.id}:${job.progress.current}`;
        if (job.progress.current > 0 && key !== lastProgressRef.current) {
          lastProgressRef.current = key;
          reloadScenes();
        }
      },
    }),
    [reloadScenes],
  );

  /** Shared error path for job-backed steps. A cancel is not an error. */
  const failStep = useCallback(
    (err: unknown, setStatus: (s: StepStatus) => void) => {
      if (err instanceof JobCancelledError) {
        setError(null);
        setStatus("idle");
        reloadScenes();
        return;
      }
      const msg = err instanceof ApiError ? err.detail : String(err);
      setError(msg);
      setStatus("error");
    },
    [reloadScenes],
  );

  /** Run one job-backed step: status bookkeeping, progress, errors, cleanup. */
  const runJobStep = useCallback(
    async <T,>(
      setStatus: (s: StepStatus) => void,
      call: (callbacks: JobCallbacks) => Promise<T>,
      onDone: (result: T) => Promise<void> | void,
    ) => {
      setStatus("running");
      setError(null);
      localJobRef.current = true;
      try {
        const result = await call(jobCallbacks);
        await onDone(result);
      } catch (err) {
        failStep(err, setStatus);
      } finally {
        localJobRef.current = false;
        setActiveJob(null);
      }
    },
    [jobCallbacks, failStep],
  );

  const cancelActiveJob = useCallback(async () => {
    if (!activeJob) return;
    try {
      const updated = await apiCancelJob(activeJob.id);
      // Show "Cancelling…" right away instead of waiting for the next poll.
      setActiveJob((cur) => (cur && cur.id === updated.id && !updated.finished_at ? updated : cur));
    } catch (err) {
      // Already finished between poll and click: nothing to cancel.
      if (!(err instanceof ApiError && err.code === "job_not_cancellable")) {
        setError(err instanceof ApiError ? err.detail : String(err));
      }
    }
  }, [activeJob]);

  const hasAudio = Boolean(config?.song.audio_file);
  const hasLyrics = Boolean(config?.song.lyrics_file);
  const analyzed = Boolean(config?.song.analyzed) || Boolean(analysisResult?.analyzed);
  const sceneCount = scenes.length;
  const descriptionsRemaining = scenes.filter(
    (s) => !s.image_prompt && !s.image_prompt_user_override,
  ).length;
  const imagesRemaining = scenes.filter((s) => !s.reference_image).length;
  const videoDescriptionsRemaining = scenes.filter(
    (s) => !s.video_prompt && !s.video_prompt_user_override,
  ).length;
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
  const upscaledCount = scenes.filter((s) => !!s.upscaled_clip).length;
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
    if (descriptionsRemaining > 0 && !hasVideos) return "descriptions";
    if (imagesRemaining > 0 && !hasVideos) return "images";
    if (videoDescriptionsRemaining > 0 && !hasVideos) return "video_descriptions";
    if (!hasVideos) return "videos";
    if (videosRemaining > 0) return "videos";
    if (upscaleRemaining > 0) return "upscale";
    return "assembly";
  }, [hasAudio, analyzed, sceneCount, descriptionsRemaining, imagesRemaining, videoDescriptionsRemaining, videosRemaining, upscaleRemaining, hasVideos]);

  const isRunning =
    uploadStatus === "running" ||
    analyzeStatus === "running" ||
    scenesStatus === "running" ||
    intakeStatus === "running" ||
    descriptionsStatus === "running" ||
    imagesStatus === "running" ||
    videoDescriptionsStatus === "running" ||
    videosStatus === "running" ||
    upscaleStatus === "running" ||
    assembleStatus === "running" ||
    activeJob !== null;

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

  const [importMessage, setImportMessage] = useState<string | null>(null);

  const importAudio = useCallback(
    async (path: string) => {
      setUploadStatus("running");
      setError(null);
      setImportMessage(null);
      try {
        const result = await apiImportAudio(path);
        await reloadConfig();
        setUploadStatus("done");
        if (result.acestep_imported) {
          const parts: string[] = ["Imported audio with AceStep metadata"];
          if (result.bpm) parts.push(`BPM ${result.bpm}`);
          if (result.keyscale) parts.push(result.keyscale);
          if (result.has_lyrics) parts.push("lyrics");
          setImportMessage(parts.join(" · "));
        } else {
          setImportMessage("Imported audio");
        }
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
    (opts?: { skipTranscription?: boolean }) =>
      runJobStep(
        setAnalyzeStatus,
        (cb) => apiRunAnalyze(opts, cb),
        async (result) => {
          setAnalysisResult(result);
          await reloadConfig();
          setAnalyzeStatus("done");
        },
      ),
    [runJobStep, reloadConfig],
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

  const [scenesMessage, setScenesMessage] = useState<string>("");

  // Phase 2: Create scenes from manual boundaries
  const confirmScenes = useCallback(
    async (
      boundaries: SceneBoundary[],
      snapToBeats: boolean,
      lyricsAssignments?: Array<{ line: string; scene_indices: number[] }>,
    ) => {
      setScenesStatus("running");
      setScenesMessage("");
      setError(null);
      try {
        const result = await apiCreateScenes(boundaries, snapToBeats, lyricsAssignments);
        await reloadScenes();
        setScenesStatus("done");
        const sourceLabels: Record<LyricsSource, string> = {
          per_scene_whisper: "Lyrics from per-scene Whisper transcription",
          word_timestamps: "Lyrics from word timestamps (may drift on vocals)",
          bpm_estimate: "Lyrics estimated from BPM (approximate)",
          manual_assignments: "Lyrics from manual line assignments",
        };
        const src = result.lyrics_source as LyricsSource | undefined;
        setScenesMessage(src ? sourceLabels[src] ?? "" : "");
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
    (opts?: { useLlm?: boolean; skipTranscription?: boolean }) =>
      runJobStep(
        setIntakeStatus,
        (cb) => apiRunIntake(opts, cb),
        async () => {
          await reloadScenes();
          await reloadConfig();
          setIntakeStatus("done");
        },
      ),
    [runJobStep, reloadScenes, reloadConfig],
  );

  /** Finish a batch step: record the result and surface per-scene failures. */
  const finishBatch = useCallback(
    async (result: BatchGenResult, setStatus: (s: StepStatus) => void, noun: string) => {
      setLastResult(result);
      await reloadScenes();
      setStatus(result.failed.length > 0 ? "error" : "done");
      if (result.failed.length > 0) {
        setError(`${result.failed.length} ${noun}(s) failed: ${result.failed[0]?.error}`);
      }
    },
    [reloadScenes],
  );

  const generateDescriptions = useCallback(
    (sceneIds?: string[]) =>
      runJobStep(
        setDescriptionsStatus,
        (cb) => apiGenerateDescriptions(sceneIds, cb),
        (result) => finishBatch(result, setDescriptionsStatus, "description"),
      ),
    [runJobStep, finishBatch],
  );

  const generateImages = useCallback(
    () =>
      runJobStep(
        setImagesStatus,
        (cb) => apiGenerateImages(undefined, undefined, cb),
        (result) => finishBatch(result, setImagesStatus, "image"),
      ),
    [runJobStep, finishBatch],
  );

  const generateVideoDescriptions = useCallback(
    (sceneIds?: string[]) =>
      runJobStep(
        setVideoDescriptionsStatus,
        (cb) => apiGenerateVideoDescriptions(sceneIds, cb),
        (result) => finishBatch(result, setVideoDescriptionsStatus, "video description"),
      ),
    [runJobStep, finishBatch],
  );

  const generateVideos = useCallback(
    (renderMode?: RenderMode) => {
      // Determine target scenes: all remaining, or unapproved for re-render
      const targetIds = videosRemaining > 0
        ? undefined
        : unapprovedSceneIds.length > 0
          ? unapprovedSceneIds
          : undefined;
      return runJobStep(
        setVideosStatus,
        (cb) => apiGenerateVideos(targetIds, undefined, renderMode, cb),
        (result) => finishBatch(result, setVideosStatus, "video"),
      );
    },
    [videosRemaining, unapprovedSceneIds, runJobStep, finishBatch],
  );

  const upscaleAll = useCallback(
    (sceneIds?: string[], resolution?: TargetResolution, upscaler?: UpscalerType, renderMode?: RenderMode) =>
      runJobStep(
        setUpscaleStatus,
        (cb) => apiUpscaleVideos(sceneIds, resolution, upscaler, renderMode, cb),
        async (result) => {
          await reloadScenes();
          setUpscaleStatus(result.failed.length > 0 ? "error" : "done");
          if (result.failed.length > 0) {
            setError(`${result.failed.length} upscale(s) failed`);
          }
        },
      ),
    [runJobStep, reloadScenes],
  );

  const assemble = useCallback(
    (approvedOnly?: boolean) =>
      runJobStep(
        setAssembleStatus,
        (cb) => apiAssemble(approvedOnly, cb),
        (result) => {
          setAssembleResult(result);
          setAssembleStatus("done");
        },
      ),
    [runJobStep],
  );

  // Show jobs this tab didn't start (an agent, another tab, or a job still
  // running after a page reload) so the UI reflects the server's queue and
  // doesn't start conflicting work. Refresh when such a job finishes.
  const projectLoaded = config !== null;
  useEffect(() => {
    if (!projectLoaded) return;
    let stopped = false;
    let watching = false;
    const tick = async () => {
      if (localJobRef.current) return;
      try {
        const jobs = await apiListJobs({ limit: 10 });
        if (stopped || localJobRef.current) return;
        const live = jobs.find((j) => j.state === "running") ?? jobs.find((j) => j.state === "queued") ?? null;
        if (live) {
          watching = true;
          jobCallbacks.onUpdate?.(live);
        } else if (watching) {
          watching = false;
          setActiveJob(null);
          await reloadScenes();
          await reloadConfig();
        }
      } catch {
        // Server unreachable or project closing; try again next tick.
      }
    };
    tick();
    const handle = setInterval(tick, 3000);
    return () => {
      stopped = true;
      clearInterval(handle);
    };
  }, [projectLoaded, jobCallbacks, reloadScenes, reloadConfig]);

  const pipelineState: PipelineState = {
    stage,
    uploadStatus,
    analyzeStatus,
    scenesStatus,
    intakeStatus,
    descriptionsStatus,
    descriptionsRemaining,
    imagesStatus,
    videoDescriptionsStatus,
    videoDescriptionsRemaining,
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
    upscaledCount,
    isRunning,
    isSegmenting: scenesStatus === "running",
    batchDone,
    batchTotal,
    activeJob,
  };

  return {
    ...pipelineState,
    cancelActiveJob,
    uploadAudio,
    uploadLyrics,
    importAudio,
    importMessage,
    importLyrics,
    runAnalyze,
    confirmScenes,
    scenesMessage,
    runAutoSegment,
    runIntake,
    generateDescriptions,
    generateImages,
    generateVideoDescriptions,
    generateVideos,
    upscaleAll,
    assemble,
  };
}
