import type {
  AnalysisResult,
  AssembleResult,
  BatchGenResult,
  FilesystemListResult,
  ImageModelType,
  ImportAudioResult,
  ImportLyricsResult,
  IntakeResult,
  Job,
  ProjectConfig,
  RegenerateImageRequest,
  RegenerateVideoRequest,
  RenderMode,
  Scene,
  SceneBoundary,
  StyleSheet,
  TargetResolution,
  UpdateSceneRequest,
  UpscaleResult,
  UpscalerType,
  VideoEngineType,
  VideoType,
} from "./types";

class ApiError extends Error {
  constructor(
    public status: number,
    public detail: string,
    public code?: string,
  ) {
    super(detail);
  }
}

/** Thrown by a job-backed call when the job ends as cancelled. */
class JobCancelledError extends Error {
  constructor(public job: Job) {
    super("Cancelled");
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    // Structured errors (spec §4.3): {"error": {code, message, detail}}.
    // Older endpoints still return FastAPI's {"detail": ...}.
    if (body?.error?.message) {
      throw new ApiError(res.status, body.error.message, body.error.code);
    }
    throw new ApiError(res.status, body.detail ?? res.statusText);
  }
  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Jobs: long-running endpoints return 202 + a Job; poll until terminal.
// ---------------------------------------------------------------------------

export interface JobCallbacks {
  /** Called with every polled snapshot (first call is the queued job). */
  onUpdate?: (job: Job) => void;
}

const JOB_POLL_MS = 1500;

export async function getJob(jobId: string): Promise<Job> {
  return request(`/api/jobs/${jobId}`);
}

export async function listJobs(opts?: { state?: string; kind?: string; limit?: number }): Promise<Job[]> {
  const params = new URLSearchParams();
  if (opts?.state) params.set("state", opts.state);
  if (opts?.kind) params.set("kind", opts.kind);
  if (opts?.limit) params.set("limit", String(opts.limit));
  const qs = params.toString();
  return request(`/api/jobs${qs ? `?${qs}` : ""}`);
}

export async function cancelJob(jobId: string): Promise<Job> {
  return request(`/api/jobs/${jobId}/cancel`, { method: "POST" });
}

/** Poll a job until it reaches a terminal state; resolve with its result. */
export async function waitForJob<T>(job: Job, callbacks?: JobCallbacks): Promise<T> {
  let current = job;
  callbacks?.onUpdate?.(current);
  while (current.state === "queued" || current.state === "running") {
    await new Promise((r) => setTimeout(r, JOB_POLL_MS));
    current = await getJob(current.id);
    callbacks?.onUpdate?.(current);
  }
  if (current.state === "succeeded") return current.result as T;
  if (current.state === "cancelled") throw new JobCancelledError(current);
  throw new ApiError(500, current.error?.message ?? "Job failed", current.error?.code);
}

/** Submit a job-shaped endpoint and wait for its result. */
async function runJob<T>(path: string, options: RequestInit, callbacks?: JobCallbacks): Promise<T> {
  const job = await request<Job>(path, options);
  return waitForJob<T>(job, callbacks);
}

export async function getConfig(): Promise<ProjectConfig> {
  return request("/api/projects/config");
}

export async function createProject(
  name: string,
  directory: string,
): Promise<{ status: string; name: string; directory: string }> {
  return request("/api/projects/create", {
    method: "POST",
    body: JSON.stringify({ name, directory }),
  });
}

export async function openProject(
  directory: string,
): Promise<{ status: string; name: string }> {
  return request("/api/projects/open", {
    method: "POST",
    body: JSON.stringify({ directory }),
  });
}

export async function closeProject(): Promise<{ status: string }> {
  return request("/api/projects/close", { method: "POST" });
}

export async function getScenes(): Promise<Scene[]> {
  return request("/api/scenes");
}

export async function updateScene(
  sceneId: string,
  updates: UpdateSceneRequest,
): Promise<Scene> {
  return request(`/api/scenes/${sceneId}`, {
    method: "PATCH",
    body: JSON.stringify(updates),
  });
}

export async function approveAll(): Promise<{ status: string; count: number }> {
  return request("/api/scenes/approve-all", { method: "POST" });
}

export async function describeImage(sceneId: string): Promise<Scene> {
  return request(`/api/scenes/${sceneId}/describe-image`, { method: "POST" });
}

export async function describeVideo(sceneId: string): Promise<Scene> {
  return request(`/api/scenes/${sceneId}/describe-video`, { method: "POST" });
}

export async function generateVideoDescriptions(
  sceneIds?: string[],
  callbacks?: JobCallbacks,
): Promise<BatchGenResult> {
  return runJob("/api/pipeline/generate-video-descriptions", {
    method: "POST",
    body: JSON.stringify({ scene_ids: sceneIds ?? [] }),
  }, callbacks);
}

export async function regenerateImage(
  sceneId: string,
  req: RegenerateImageRequest,
  callbacks?: JobCallbacks,
): Promise<Scene> {
  const result = await runJob<{ scene: Scene }>(`/api/scenes/${sceneId}/regenerate-image`, {
    method: "POST",
    body: JSON.stringify(req),
  }, callbacks);
  return result.scene;
}

export async function regenerateVideo(
  sceneId: string,
  req: RegenerateVideoRequest,
  callbacks?: JobCallbacks,
): Promise<Scene> {
  const result = await runJob<{ scene: Scene }>(`/api/scenes/${sceneId}/regenerate-video`, {
    method: "POST",
    body: JSON.stringify(req),
  }, callbacks);
  return result.scene;
}

export async function assemblePreview(
  approvedOnly?: boolean,
  callbacks?: JobCallbacks,
): Promise<AssembleResult> {
  return runJob("/api/pipeline/assemble", {
    method: "POST",
    body: JSON.stringify({ approved_only: approvedOnly ?? false }),
  }, callbacks);
}

export async function listFilesystem(
  path?: string,
  type?: string,
): Promise<FilesystemListResult> {
  const params = new URLSearchParams();
  if (path) params.set("path", path);
  if (type) params.set("type", type);
  const qs = params.toString();
  return request(`/api/filesystem/list${qs ? `?${qs}` : ""}`);
}

export async function importAudio(
  path: string,
): Promise<ImportAudioResult> {
  return request("/api/import/audio", {
    method: "POST",
    body: JSON.stringify({ path }),
  });
}

export async function importLyrics(
  path: string,
): Promise<ImportLyricsResult> {
  return request("/api/import/lyrics", {
    method: "POST",
    body: JSON.stringify({ path }),
  });
}

export async function uploadAudio(
  file: File,
): Promise<{ status: string; path: string }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/upload/audio", { method: "POST", body: form });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail ?? res.statusText);
  }
  return res.json();
}

export async function uploadLyrics(
  file: File,
): Promise<{ status: string; path: string }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/upload/lyrics", { method: "POST", body: form });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail ?? res.statusText);
  }
  return res.json();
}

export async function runAnalyze(opts?: {
  skipTranscription?: boolean;
  useVocalSeparation?: boolean;
}, callbacks?: JobCallbacks): Promise<AnalysisResult> {
  const params = new URLSearchParams();
  if (opts?.skipTranscription !== undefined)
    params.set("skip_transcription", String(opts.skipTranscription));
  if (opts?.useVocalSeparation !== undefined)
    params.set("use_vocal_separation", String(opts.useVocalSeparation));
  const qs = params.toString();
  return runJob(`/api/pipeline/analyze${qs ? `?${qs}` : ""}`, {
    method: "POST",
  }, callbacks);
}

export async function getAnalysis(): Promise<AnalysisResult> {
  return request("/api/analysis");
}

export async function createScenes(
  boundaries: SceneBoundary[],
  snapToBeats: boolean = false,
  lyricsAssignments?: Array<{ line: string; scene_indices: number[] }>,
): Promise<IntakeResult> {
  return request("/api/pipeline/create-scenes", {
    method: "POST",
    body: JSON.stringify({
      boundaries,
      snap_to_beats: snapToBeats,
      lyrics_assignments: lyricsAssignments ?? null,
    }),
  });
}

export async function autoSegment(
  useLlm: boolean = true,
): Promise<IntakeResult> {
  const params = new URLSearchParams();
  params.set("use_llm", String(useLlm));
  return request(`/api/pipeline/auto-segment?${params}`, {
    method: "POST",
  });
}

export async function runIntake(opts?: {
  useLlm?: boolean;
  skipTranscription?: boolean;
}, callbacks?: JobCallbacks): Promise<IntakeResult> {
  const params = new URLSearchParams();
  if (opts?.useLlm !== undefined)
    params.set("use_llm", String(opts.useLlm));
  if (opts?.skipTranscription !== undefined)
    params.set("skip_transcription", String(opts.skipTranscription));
  const qs = params.toString();
  return runJob(`/api/pipeline/intake${qs ? `?${qs}` : ""}`, {
    method: "POST",
  }, callbacks);
}

export async function generateDescriptions(
  sceneIds?: string[],
  callbacks?: JobCallbacks,
): Promise<BatchGenResult> {
  return runJob("/api/pipeline/generate-descriptions", {
    method: "POST",
    body: JSON.stringify({ scene_ids: sceneIds ?? [] }),
  }, callbacks);
}

export async function generateAllImages(
  sceneIds?: string[],
  model?: ImageModelType,
  callbacks?: JobCallbacks,
): Promise<BatchGenResult> {
  return runJob("/api/pipeline/generate-images", {
    method: "POST",
    body: JSON.stringify({ scene_ids: sceneIds ?? [], model: model ?? null }),
  }, callbacks);
}

export async function generateAllVideos(
  sceneIds?: string[],
  engine?: VideoEngineType,
  renderMode?: RenderMode,
  callbacks?: JobCallbacks,
): Promise<BatchGenResult> {
  return runJob("/api/pipeline/generate-videos", {
    method: "POST",
    body: JSON.stringify({
      scene_ids: sceneIds ?? [],
      engine: engine ?? null,
      render_mode: renderMode ?? "preview",
    }),
  }, callbacks);
}

export async function upscaleVideos(
  sceneIds?: string[],
  resolution?: TargetResolution,
  upscaler?: UpscalerType,
  renderMode?: RenderMode,
  callbacks?: JobCallbacks,
): Promise<UpscaleResult> {
  return runJob("/api/pipeline/upscale", {
    method: "POST",
    body: JSON.stringify({
      scene_ids: sceneIds ?? [],
      resolution: resolution ?? null,
      upscaler: upscaler ?? null,
      render_mode: renderMode ?? "final",
    }),
  }, callbacks);
}

export async function upscaleScene(
  sceneId: string,
  resolution?: TargetResolution,
  upscaler?: UpscalerType,
  callbacks?: JobCallbacks,
): Promise<UpscaleResult> {
  return runJob(`/api/scenes/${sceneId}/upscale`, {
    method: "POST",
    body: JSON.stringify({
      scene_ids: [sceneId],
      resolution: resolution ?? null,
      upscaler: upscaler ?? null,
    }),
  }, callbacks);
}

export async function getSegmentMarkers(): Promise<{ markers: number[] }> {
  return request("/api/segment-markers");
}

export async function saveSegmentMarkers(markers: number[]): Promise<{ status: string }> {
  return request("/api/segment-markers", {
    method: "PUT",
    body: JSON.stringify({ markers }),
  });
}

export async function getLyricsAssignments(): Promise<{ assignments: Array<{ line: string; scene_indices: number[] }> }> {
  return request("/api/lyrics-assignments");
}

export async function saveLyricsAssignments(
  assignments: Array<{ line: string; scene_indices: number[] }>,
): Promise<{ status: string }> {
  return request("/api/lyrics-assignments", {
    method: "PUT",
    body: JSON.stringify({ assignments }),
  });
}

export async function updateStyleSheet(styleSheet: StyleSheet): Promise<{ status: string }> {
  return request("/api/projects/config/style-sheet", {
    method: "PUT",
    body: JSON.stringify(styleSheet),
  });
}

export async function updateVideoType(videoType: VideoType): Promise<{ status: string; video_type: VideoType }> {
  return request("/api/projects/config/video-type", {
    method: "PUT",
    body: JSON.stringify({ video_type: videoType }),
  });
}

export function fileUrl(path: string, bustCache?: number): string {
  const base = `/files/${path}`;
  return bustCache ? `${base}?t=${bustCache}` : base;
}

export { ApiError, JobCancelledError };
