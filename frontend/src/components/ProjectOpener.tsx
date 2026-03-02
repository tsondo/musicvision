import { useState } from "react";

interface Props {
  onOpen: (directory: string) => void;
  onCreate: (name: string, directory: string) => void;
  lastPath: string;
  error?: string;
}

export default function ProjectOpener({ onOpen, onCreate, lastPath, error }: Props) {
  const [tab, setTab] = useState<"open" | "create">("open");
  const [directory, setDirectory] = useState(lastPath);
  const [newName, setNewName] = useState("");
  const [newDirectory, setNewDirectory] = useState("");

  const handleOpen = (e: React.FormEvent) => {
    e.preventDefault();
    if (directory.trim()) onOpen(directory.trim());
  };

  const handleCreate = (e: React.FormEvent) => {
    e.preventDefault();
    if (newName.trim() && newDirectory.trim())
      onCreate(newName.trim(), newDirectory.trim());
  };

  return (
    <div className="project-opener">
      <h1>MusicVision</h1>
      <p className="subtitle">AI Music Video Pipeline</p>

      <div className="tab-bar">
        <button
          className={`tab-btn ${tab === "open" ? "active" : ""}`}
          onClick={() => setTab("open")}
          type="button"
        >
          Open Existing
        </button>
        <button
          className={`tab-btn ${tab === "create" ? "active" : ""}`}
          onClick={() => setTab("create")}
          type="button"
        >
          Create New
        </button>
      </div>

      {tab === "open" ? (
        <form onSubmit={handleOpen}>
          <label htmlFor="project-path">Project directory</label>
          <div className="input-row">
            <input
              id="project-path"
              type="text"
              value={directory}
              onChange={(e) => setDirectory(e.target.value)}
              placeholder="/path/to/project"
              autoFocus
            />
            <button type="submit" disabled={!directory.trim()}>
              Open
            </button>
          </div>
        </form>
      ) : (
        <form onSubmit={handleCreate}>
          <label htmlFor="new-name">Project name</label>
          <div className="input-row" style={{ marginBottom: 12 }}>
            <input
              id="new-name"
              type="text"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="My Music Video"
              autoFocus
            />
          </div>
          <label htmlFor="new-dir">Directory</label>
          <div className="input-row">
            <input
              id="new-dir"
              type="text"
              value={newDirectory}
              onChange={(e) => setNewDirectory(e.target.value)}
              placeholder="/path/to/new/project"
            />
            <button type="submit" disabled={!newName.trim() || !newDirectory.trim()}>
              Create
            </button>
          </div>
        </form>
      )}

      {error && <p className="error">{error}</p>}
    </div>
  );
}
