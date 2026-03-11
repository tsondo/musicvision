import { useCallback, useEffect, useRef, useState } from "react";
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

interface ImageJob {
  kind: "image";
  sceneId: string;
  req: RegenerateImageRequest;
}

interface VideoJob {
  kind: "video";
  sceneId: string;
  req: RegenerateVideoRequest;
}

type QueueItem = ImageJob | VideoJob;

export type SceneGenStatus = "queued" | "running" | "idle";

export function useScenes(projectLoaded: boolean) {
  const [scenes, setScenes] = useState<Scene[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [generating, setGenerating] = useState<Set<string>>(new Set());

  // Unified generation queue (images + videos)
  const [queue, setQueue] = useState<QueueItem[]>([]);
  const [queueTotal, setQueueTotal] = useState(0);
  const processingRef = useRef(false);

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

  // --- Queue management ---

  const genKey = (sceneId: string, kind: "image" | "video") =>
    `${sceneId}:${kind}`;

  const queueImage = useCallback(
    (sceneId: string, req: RegenerateImageRequest) => {
      setQueue((prev) => {
        if (prev.some((item) => item.kind === "image" && item.sceneId === sceneId)) return prev;
        return [...prev, { kind: "image", sceneId, req }];
      });
      setQueueTotal((prev) => prev + 1);
      setGenerating((prev) => new Set(prev).add(genKey(sceneId, "image") + ":queued"));
    },
    [],
  );

  const queueVideo = useCallback(
    (sceneId: string, req: RegenerateVideoRequest) => {
      setQueue((prev) => {
        if (prev.some((item) => item.kind === "video" && item.sceneId === sceneId)) return prev;
        return [...prev, { kind: "video", sceneId, req }];
      });
      setQueueTotal((prev) => prev + 1);
      setGenerating((prev) => new Set(prev).add(genKey(sceneId, "video") + ":queued"));
    },
    [],
  );

  const dequeue = useCallback(
    (sceneId: string, kind: "image" | "video") => {
      setQueue((prev) => prev.filter(
        (item) => !(item.sceneId === sceneId && item.kind === kind),
      ));
      setQueueTotal((prev) => Math.max(0, prev - 1));
      setGenerating((prev) => {
        const next = new Set(prev);
        next.delete(genKey(sceneId, kind) + ":queued");
        return next;
      });
    },
    [],
  );

  // Process the queue one job at a time
  useEffect(() => {
    if (queue.length === 0 || processingRef.current) return;

    const processNext = async () => {
      processingRef.current = true;
      const item = queue[0]!;
      const key = genKey(item.sceneId, item.kind);

      // Move from queued to running
      setGenerating((prev) => {
        const next = new Set(prev);
        next.delete(key + ":queued");
        next.add(key);
        return next;
      });

      try {
        const updated =
          item.kind === "image"
            ? await apiRegenImage(item.sceneId, item.req)
            : await apiRegenVideo(item.sceneId, item.req);
        setScenes((prev) =>
          prev.map((s) => (s.id === item.sceneId ? updated : s)),
        );
      } catch (err) {
        console.error(`${item.kind} generation failed for ${item.sceneId}:`, err);
      } finally {
        setGenerating((prev) => {
          const next = new Set(prev);
          next.delete(key);
          return next;
        });
        setQueue((prev) => {
          const next = prev.slice(1);
          if (next.length === 0) setQueueTotal(0);
          return next;
        });
        processingRef.current = false;
      }
    };

    processNext();
  }, [queue]);

  // --- Status helpers ---

  const genStatus = useCallback(
    (sceneId: string, kind: "image" | "video"): SceneGenStatus => {
      const key = genKey(sceneId, kind);
      if (generating.has(key)) return "running";
      if (generating.has(key + ":queued")) return "queued";
      return "idle";
    },
    [generating],
  );

  const imageGenStatus = useCallback(
    (sceneId: string): SceneGenStatus => genStatus(sceneId, "image"),
    [genStatus],
  );

  const videoGenStatus = useCallback(
    (sceneId: string): SceneGenStatus => genStatus(sceneId, "video"),
    [genStatus],
  );

  const queueActive = queue.length > 0 || processingRef.current;
  // +1 for the currently processing item (already removed from queue)
  const queueRemaining = queue.length + (processingRef.current ? 1 : 0);
  const queueDone = queueTotal - queueRemaining;

  return {
    scenes,
    loading,
    error,
    queueActive,
    queueDone,
    queueTotal,
    imageGenStatus,
    videoGenStatus,
    reload: load,
    updateScene,
    regenerateImage: queueImage,
    regenerateVideo: queueVideo,
    dequeueImage: (sceneId: string) => dequeue(sceneId, "image"),
    dequeueVideo: (sceneId: string) => dequeue(sceneId, "video"),
    clearScenes: useCallback(() => setScenes([]), []),
  };
}
