import type {
  AssembleResult,
  BatchGenResult,
  FilesystemListResult,
  ImageModelType,
  ImportAudioResult,
  ImportLyricsResult,
  IntakeResult,
  ProjectConfig,
  RegenerateImageRequest,
  RegenerateVideoRequest,
  RenderMode,
  Scene,
  TargetResolution,
  UpdateSceneRequest,
  UpscaleResult,
  UpscalerType,
  VideoEngineType,
} from "./types";

class ApiError extends Error {
  constructor(
    public status: number,
    public detail: string,
  ) {
    super(detail);
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail ?? res.statusText);
  }
  return res.json() as Promise<T>;
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

export async function regenerateImage(
  sceneId: string,
  req: RegenerateImageRequest,
): Promise<Scene> {
  return request(`/api/scenes/${sceneId}/regenerate-image`, {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export async function regenerateVideo(
  sceneId: string,
  req: RegenerateVideoRequest,
): Promise<Scene> {
  return request(`/api/scenes/${sceneId}/regenerate-video`, {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export async function assemblePreview(approvedOnly?: boolean): Promise<AssembleResult> {
  return request("/api/pipeline/assemble", {
    method: "POST",
    body: JSON.stringify({ approved_only: approvedOnly ?? false }),
  });
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

export async function runIntake(opts?: {
  useLlm?: boolean;
  skipTranscription?: boolean;
}): Promise<IntakeResult> {
  const params = new URLSearchParams();
  if (opts?.useLlm !== undefined)
    params.set("use_llm", String(opts.useLlm));
  if (opts?.skipTranscription !== undefined)
    params.set("skip_transcription", String(opts.skipTranscription));
  const qs = params.toString();
  return request(`/api/pipeline/intake${qs ? `?${qs}` : ""}`, {
    method: "POST",
  });
}

export async function generateAllImages(
  sceneIds?: string[],
  model?: ImageModelType,
): Promise<BatchGenResult> {
  return request("/api/pipeline/generate-images", {
    method: "POST",
    body: JSON.stringify({ scene_ids: sceneIds ?? [], model: model ?? null }),
  });
}

export async function generateAllVideos(
  sceneIds?: string[],
  engine?: VideoEngineType,
  renderMode?: RenderMode,
): Promise<BatchGenResult> {
  return request("/api/pipeline/generate-videos", {
    method: "POST",
    body: JSON.stringify({
      scene_ids: sceneIds ?? [],
      engine: engine ?? null,
      render_mode: renderMode ?? "preview",
    }),
  });
}

export async function upscaleVideos(
  sceneIds?: string[],
  resolution?: TargetResolution,
  upscaler?: UpscalerType,
  renderMode?: RenderMode,
): Promise<UpscaleResult> {
  return request("/api/pipeline/upscale", {
    method: "POST",
    body: JSON.stringify({
      scene_ids: sceneIds ?? [],
      resolution: resolution ?? null,
      upscaler: upscaler ?? null,
      render_mode: renderMode ?? "final",
    }),
  });
}

export async function upscaleScene(
  sceneId: string,
  resolution?: TargetResolution,
  upscaler?: UpscalerType,
): Promise<UpscaleResult> {
  return request(`/api/scenes/${sceneId}/upscale`, {
    method: "POST",
    body: JSON.stringify({
      scene_ids: [sceneId],
      resolution: resolution ?? null,
      upscaler: upscaler ?? null,
    }),
  });
}

export function fileUrl(path: string, bustCache?: number): string {
  const base = `/files/${path}`;
  return bustCache ? `${base}?t=${bustCache}` : base;
}

export { ApiError };
