import type { ProjectConfig, Scene, UpdateSceneRequest } from "./types";

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

export function fileUrl(path: string): string {
  return `/files/${path}`;
}

export { ApiError };
