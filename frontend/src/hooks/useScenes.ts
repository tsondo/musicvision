import { useCallback, useEffect, useState } from "react";
import {
  getScenes,
  updateScene as apiUpdateScene,
  regenerateImage as apiRegenImage,
  regenerateVideo as apiRegenVideo,
} from "../api/client";
import type {
  RegenerateImageRequest,
  RegenerateVideoRequest,
  Scene,
  UpdateSceneRequest,
} from "../api/types";

export function useScenes(projectLoaded: boolean) {
  const [scenes, setScenes] = useState<Scene[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [generating, setGenerating] = useState<Set<string>>(new Set());

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getScenes();
      setScenes(data);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (projectLoaded) load();
  }, [projectLoaded, load]);

  const updateScene = useCallback(
    async (sceneId: string, updates: UpdateSceneRequest) => {
      const updated = await apiUpdateScene(sceneId, updates);
      setScenes((prev) =>
        prev.map((s) => (s.id === sceneId ? updated : s)),
      );
      return updated;
    },
    [],
  );

  const regenerateImage = useCallback(
    async (sceneId: string, req: RegenerateImageRequest) => {
      setGenerating((prev) => new Set(prev).add(sceneId + ":image"));
      try {
        const updated = await apiRegenImage(sceneId, req);
        setScenes((prev) =>
          prev.map((s) => (s.id === sceneId ? updated : s)),
        );
        return updated;
      } finally {
        setGenerating((prev) => {
          const next = new Set(prev);
          next.delete(sceneId + ":image");
          return next;
        });
      }
    },
    [],
  );

  const regenerateVideo = useCallback(
    async (sceneId: string, req: RegenerateVideoRequest) => {
      setGenerating((prev) => new Set(prev).add(sceneId + ":video"));
      try {
        const updated = await apiRegenVideo(sceneId, req);
        setScenes((prev) =>
          prev.map((s) => (s.id === sceneId ? updated : s)),
        );
        return updated;
      } finally {
        setGenerating((prev) => {
          const next = new Set(prev);
          next.delete(sceneId + ":video");
          return next;
        });
      }
    },
    [],
  );

  return {
    scenes,
    loading,
    error,
    generating,
    reload: load,
    updateScene,
    regenerateImage,
    regenerateVideo,
  };
}
