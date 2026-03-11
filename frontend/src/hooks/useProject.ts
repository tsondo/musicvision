import { useCallback, useEffect, useState } from "react";
import { closeProject as apiCloseProject, createProject, getConfig, openProject, ApiError } from "../api/client";
import type { ProjectConfig } from "../api/types";

export type ProjectState =
  | { status: "no-project" }
  | { status: "loading" }
  | { status: "loaded"; config: ProjectConfig }
  | { status: "error"; message: string };

const LAST_PROJECT_KEY = "musicvision_last_project";

export function useProject() {
  const [state, setState] = useState<ProjectState>({ status: "loading" });

  const tryLoadConfig = useCallback(async () => {
    setState({ status: "loading" });
    try {
      const config = await getConfig();
      setState({ status: "loaded", config });
    } catch (err) {
      if (err instanceof ApiError && err.status === 400) {
        setState({ status: "no-project" });
      } else {
        setState({ status: "error", message: String(err) });
      }
    }
  }, []);

  const open = useCallback(async (directory: string) => {
    setState({ status: "loading" });
    try {
      await openProject(directory);
      localStorage.setItem(LAST_PROJECT_KEY, directory);
      const config = await getConfig();
      setState({ status: "loaded", config });
    } catch (err) {
      const msg = err instanceof ApiError ? err.detail : String(err);
      setState({ status: "error", message: msg });
    }
  }, []);

  const create = useCallback(async (name: string, directory: string) => {
    setState({ status: "loading" });
    try {
      await createProject(name, directory);
      localStorage.setItem(LAST_PROJECT_KEY, directory);
      const config = await getConfig();
      setState({ status: "loaded", config });
    } catch (err) {
      const msg = err instanceof ApiError ? err.detail : String(err);
      setState({ status: "error", message: msg });
    }
  }, []);

  useEffect(() => {
    tryLoadConfig();
  }, [tryLoadConfig]);

  const lastProjectPath = localStorage.getItem(LAST_PROJECT_KEY) ?? "";

  const close = useCallback(() => {
    localStorage.removeItem(LAST_PROJECT_KEY);
    apiCloseProject().catch(() => {});
    setState({ status: "no-project" });
  }, []);

  const updateConfig = useCallback((config: ProjectConfig) => {
    setState({ status: "loaded", config });
  }, []);

  return { state, open, create, close, reload: tryLoadConfig, lastProjectPath, updateConfig };
}
