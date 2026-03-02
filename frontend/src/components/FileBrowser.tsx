import { useCallback, useEffect, useState } from "react";
import { listFilesystem, ApiError } from "../api/client";
import type { FilesystemEntry } from "../api/types";

interface Props {
  mode: "directory" | "file";
  fileFilter?: string[];
  startPath?: string;
  title: string;
  onSelect: (path: string) => void;
  onCancel: () => void;
}

export default function FileBrowser({
  mode,
  fileFilter,
  startPath,
  title,
  onSelect,
  onCancel,
}: Props) {
  const [currentPath, setCurrentPath] = useState(startPath || "");
  const [entries, setEntries] = useState<FilesystemEntry[]>([]);
  const [parentPath, setParentPath] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const loadDirectory = useCallback(
    async (path: string) => {
      setLoading(true);
      setError(null);
      setSelectedFile(null);
      try {
        const listType = mode === "directory" ? "directory" : "all";
        const result = await listFilesystem(path || undefined, listType);
        let filtered = result.entries;
        // In file mode with a filter, show directories + matching files
        if (mode === "file" && fileFilter && fileFilter.length > 0) {
          filtered = result.entries.filter(
            (e) =>
              e.is_dir ||
              fileFilter.some((ext) => e.name.toLowerCase().endsWith(ext.toLowerCase())),
          );
        }
        setEntries(filtered);
        setParentPath(result.parent);
        // Update currentPath to the resolved path from the first entry's parent
        if (filtered.length > 0 && filtered[0]) {
          const resolvedDir = filtered[0].path.substring(0, filtered[0].path.lastIndexOf("/"));
          setCurrentPath(resolvedDir);
        } else if (result.parent !== null) {
          // Empty directory — derive from parent
          setCurrentPath(path);
        }
      } catch (err) {
        const msg = err instanceof ApiError ? err.detail : String(err);
        setError(msg);
      } finally {
        setLoading(false);
      }
    },
    [mode, fileFilter],
  );

  useEffect(() => {
    loadDirectory(startPath || "");
  }, [loadDirectory, startPath]);

  const handleEntryClick = (entry: FilesystemEntry) => {
    if (entry.is_dir) {
      loadDirectory(entry.path);
    } else {
      setSelectedFile(entry.path);
    }
  };

  const handleGoUp = () => {
    if (parentPath) loadDirectory(parentPath);
  };

  const handleBreadcrumbClick = (path: string) => {
    loadDirectory(path);
  };

  const handleConfirm = () => {
    if (mode === "directory") {
      onSelect(currentPath);
    } else if (selectedFile) {
      onSelect(selectedFile);
    }
  };

  // Build breadcrumb segments from currentPath
  const breadcrumbs: { label: string; path: string }[] = [];
  if (currentPath) {
    const parts = currentPath.split("/").filter(Boolean);
    let accumulated = "";
    for (const part of parts) {
      accumulated += "/" + part;
      breadcrumbs.push({ label: part, path: accumulated });
    }
  }

  const canConfirm = mode === "directory" || selectedFile !== null;

  return (
    <div className="file-browser-overlay" onClick={onCancel}>
      <div className="file-browser" onClick={(e) => e.stopPropagation()}>
        <div className="fb-header">
          <span className="fb-title">{title}</span>
          <button className="btn-sm btn-secondary" onClick={onCancel}>
            Esc
          </button>
        </div>

        <div className="fb-breadcrumbs">
          <span
            className="fb-crumb"
            onClick={() => handleBreadcrumbClick("")}
          >
            ~
          </span>
          {breadcrumbs.map((bc) => (
            <span key={bc.path}>
              <span className="fb-sep">/</span>
              <span
                className="fb-crumb"
                onClick={() => handleBreadcrumbClick(bc.path)}
              >
                {bc.label}
              </span>
            </span>
          ))}
        </div>

        {error && <div className="fb-error">{error}</div>}

        <div className="fb-entries">
          {loading ? (
            <div className="fb-empty">Loading...</div>
          ) : (
            <>
              {parentPath && (
                <div className="fb-entry is-dir" onClick={handleGoUp}>
                  ..
                </div>
              )}
              {entries.length === 0 && !parentPath ? (
                <div className="fb-empty">Empty directory</div>
              ) : (
                entries.map((entry) => (
                  <div
                    key={entry.path}
                    className={`fb-entry${entry.is_dir ? " is-dir" : ""}${selectedFile === entry.path ? " selected" : ""}`}
                    onClick={() => handleEntryClick(entry)}
                    onDoubleClick={() => {
                      if (entry.is_dir) {
                        loadDirectory(entry.path);
                      } else {
                        onSelect(entry.path);
                      }
                    }}
                  >
                    {entry.is_dir ? `${entry.name}/` : entry.name}
                  </div>
                ))
              )}
            </>
          )}
        </div>

        {mode === "directory" && currentPath && (
          <div className="fb-selected-path">{currentPath}</div>
        )}

        <div className="fb-footer">
          <button className="btn-sm btn-secondary" onClick={onCancel}>
            Cancel
          </button>
          <button className="btn-sm" disabled={!canConfirm} onClick={handleConfirm}>
            Select
          </button>
        </div>
      </div>
    </div>
  );
}
