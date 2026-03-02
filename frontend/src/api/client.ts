import type {
  BatchGenResult,
  IntakeResult,
  ProjectConfig,
  RegenerateImageRequest,
  RegenerateVideoRequest,
  Scene,
  UpdateSceneRequest,
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

export async function assemblePreview(): Promise<{
  status: string;
  rough_cut: string;
}> {
  return request("/api/pipeline/assemble", {
    method: "POST",
    body: JSON.stringify({}),
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
): Promise<BatchGenResult> {
  return request("/api/pipeline/generate-images", {
    method: "POST",
    body: JSON.stringify({ scene_ids: sceneIds ?? [] }),
  });
}

export async function generateAllVideos(
  sceneIds?: string[],
): Promise<BatchGenResult> {
  return request("/api/pipeline/generate-videos", {
    method: "POST",
    body: JSON.stringify({ scene_ids: sceneIds ?? [] }),
  });
}

export function fileUrl(path: string): string {
  return `/files/${path}`;
}

export { ApiError };
