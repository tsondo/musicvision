import { useState } from "react";

interface Props {
  onOpen: (directory: string) => void;
  lastPath: string;
  error?: string;
}

export default function ProjectOpener({ onOpen, lastPath, error }: Props) {
  const [directory, setDirectory] = useState(lastPath);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (directory.trim()) onOpen(directory.trim());
  };

  return (
    <div className="project-opener">
      <h1>MusicVision</h1>
      <p className="subtitle">Scene Review</p>
      <form onSubmit={handleSubmit}>
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
      {error && <p className="error">{error}</p>}
    </div>
  );
}
