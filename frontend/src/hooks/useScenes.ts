import { useCallback, useEffect, useState } from "react";
import { getScenes, updateScene as apiUpdateScene, approveAll as apiApproveAll } from "../api/client";
import type { Scene, UpdateSceneRequest } from "../api/types";

export function useScenes(projectLoaded: boolean) {
  const [scenes, setScenes] = useState<Scene[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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

  const approveAll = useCallback(async () => {
    await apiApproveAll();
    await load();
  }, [load]);

  return { scenes, loading, error, reload: load, updateScene, approveAll };
}
