import { useState } from "react";
import type { Scene, UpdateSceneRequest, ApprovalStatus } from "../api/types";
import { fileUrl } from "../api/client";
import AudioPlayer from "./AudioPlayer";

interface Props {
  scene: Scene;
  onUpdate: (sceneId: string, updates: UpdateSceneRequest) => Promise<Scene>;
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = (seconds % 60).toFixed(1);
  return `${m}:${s.padStart(4, "0")}`;
}

export default function SceneCard({ scene, onUpdate }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [editingField, setEditingField] = useState<
    "image_prompt" | "video_prompt" | "notes" | null
  >(null);
  const [editValue, setEditValue] = useState("");
  const [saving, setSaving] = useState(false);

  const duration = scene.time_end - scene.time_start;
  const effectiveImagePrompt =
    scene.image_prompt_user_override || scene.image_prompt;
  const effectiveVideoPrompt =
    scene.video_prompt_user_override || scene.video_prompt;

  const startEdit = (
    field: "image_prompt" | "video_prompt" | "notes",
  ) => {
    if (field === "image_prompt") {
      setEditValue(scene.image_prompt_user_override || scene.image_prompt || "");
    } else if (field === "video_prompt") {
      setEditValue(scene.video_prompt_user_override || scene.video_prompt || "");
    } else {
      setEditValue(scene.notes);
    }
    setEditingField(field);
  };

  const saveEdit = async () => {
    if (!editingField) return;
    setSaving(true);
    const updates: UpdateSceneRequest = {};
    if (editingField === "image_prompt") {
      updates.image_prompt_user_override = editValue;
    } else if (editingField === "video_prompt") {
      updates.video_prompt_user_override = editValue;
    } else {
      updates.notes = editValue;
    }
    try {
      await onUpdate(scene.id, updates);
    } finally {
      setSaving(false);
      setEditingField(null);
    }
  };

  const cancelEdit = () => setEditingField(null);

  const setStatus = async (
    type: "image" | "video",
    status: ApprovalStatus,
  ) => {
    const updates: UpdateSceneRequest =
      type === "image" ? { image_status: status } : { video_status: status };
    await onUpdate(scene.id, updates);
  };

  return (
    <div className={`scene-card status-${scene.image_status}`}>
      {/* Header */}
      <div className="scene-header">
        <span className="scene-id">{scene.id}</span>
        <span className="scene-time">
          {formatTime(scene.time_start)} - {formatTime(scene.time_end)}
        </span>
        <span className="scene-duration">{duration.toFixed(1)}s</span>
        <span className={`scene-type badge-${scene.type}`}>{scene.type}</span>
      </div>

      {/* Image */}
      <div className="scene-image">
        {scene.reference_image ? (
          <img
            src={fileUrl(scene.reference_image)}
            alt={`Scene ${scene.id}`}
            loading="lazy"
          />
        ) : (
          <div className="image-placeholder">No image</div>
        )}
      </div>

      {/* Audio */}
      <AudioPlayer path={scene.audio_segment} />

      {/* Lyrics */}
      {scene.lyrics && (
        <div
          className={`scene-lyrics ${expanded ? "expanded" : ""}`}
          onClick={() => setExpanded(!expanded)}
        >
          {scene.lyrics}
        </div>
      )}

      {/* Prompts */}
      <div className="scene-prompts">
        {editingField === "image_prompt" ? (
          <div className="edit-area">
            <label>Image prompt</label>
            <textarea
              value={editValue}
              onChange={(e) => setEditValue(e.target.value)}
              rows={4}
              autoFocus
            />
            <div className="edit-actions">
              <button onClick={saveEdit} disabled={saving}>
                Save
              </button>
              <button onClick={cancelEdit} className="btn-secondary">
                Cancel
              </button>
            </div>
          </div>
        ) : (
          effectiveImagePrompt && (
            <div
              className="prompt-text"
              onClick={() => startEdit("image_prompt")}
              title="Click to edit"
            >
              <span className="prompt-label">IMG</span>
              {effectiveImagePrompt.length > 120 && !expanded
                ? effectiveImagePrompt.slice(0, 120) + "..."
                : effectiveImagePrompt}
            </div>
          )
        )}

        {editingField === "video_prompt" ? (
          <div className="edit-area">
            <label>Video prompt</label>
            <textarea
              value={editValue}
              onChange={(e) => setEditValue(e.target.value)}
              rows={4}
              autoFocus
            />
            <div className="edit-actions">
              <button onClick={saveEdit} disabled={saving}>
                Save
              </button>
              <button onClick={cancelEdit} className="btn-secondary">
                Cancel
              </button>
            </div>
          </div>
        ) : (
          effectiveVideoPrompt && (
            <div
              className="prompt-text"
              onClick={() => startEdit("video_prompt")}
              title="Click to edit"
            >
              <span className="prompt-label">VID</span>
              {effectiveVideoPrompt.length > 120 && !expanded
                ? effectiveVideoPrompt.slice(0, 120) + "..."
                : effectiveVideoPrompt}
            </div>
          )
        )}
      </div>

      {/* Notes */}
      {editingField === "notes" ? (
        <div className="edit-area">
          <label>Notes</label>
          <textarea
            value={editValue}
            onChange={(e) => setEditValue(e.target.value)}
            rows={2}
            autoFocus
          />
          <div className="edit-actions">
            <button onClick={saveEdit} disabled={saving}>
              Save
            </button>
            <button onClick={cancelEdit} className="btn-secondary">
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <div
          className="scene-notes"
          onClick={() => startEdit("notes")}
          title="Click to edit notes"
        >
          {scene.notes || "Add notes..."}
        </div>
      )}

      {/* Sub-clips */}
      {scene.sub_clips.length > 0 && (
        <div className="sub-clips">
          <span className="sub-clips-label">
            {scene.sub_clips.length} sub-clips
          </span>
          <div className="sub-clips-list">
            {scene.sub_clips.map((sc) => (
              <span
                key={sc.id}
                className={`sub-clip-badge ${sc.video_clip ? "has-clip" : ""}`}
              >
                {sc.id.split("_").pop()}
                {" "}
                ({(sc.time_end - sc.time_start).toFixed(1)}s)
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Approval buttons */}
      <div className="scene-actions">
        <button
          className={`btn-approve ${scene.image_status === "approved" ? "active" : ""}`}
          onClick={() =>
            setStatus(
              "image",
              scene.image_status === "approved" ? "pending" : "approved",
            )
          }
          disabled={!scene.reference_image}
        >
          {scene.image_status === "approved" ? "Approved" : "Approve"}
        </button>
        <button
          className={`btn-reject ${scene.image_status === "rejected" ? "active" : ""}`}
          onClick={() =>
            setStatus(
              "image",
              scene.image_status === "rejected" ? "pending" : "rejected",
            )
          }
          disabled={!scene.reference_image}
        >
          Reject
        </button>
      </div>
    </div>
  );
}
