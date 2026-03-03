import { useState } from "react";
import { assemblePreview, fileUrl } from "../api/client";

export default function PreviewPanel() {
  const [assembling, setAssembling] = useState(false);
  const [roughCut, setRoughCut] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleAssemble = async () => {
    setAssembling(true);
    setError(null);
    try {
      const result = await assemblePreview();
      setRoughCut(result.rough_cut);
    } catch (err) {
      setError(String(err));
    } finally {
      setAssembling(false);
    }
  };

  return (
    <div className="preview-panel">
      <button
        onClick={handleAssemble}
        disabled={assembling}
        className="btn-preview"
      >
        {assembling ? "Stitching..." : "Stitch and Mux"}
      </button>
      {roughCut && (
        <a
          href={fileUrl(roughCut)}
          target="_blank"
          rel="noopener noreferrer"
          className="preview-link"
        >
          rough_cut.mp4
        </a>
      )}
      {error && <span className="preview-error">{error}</span>}
    </div>
  );
}
