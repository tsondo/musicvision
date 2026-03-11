import { useCallback, useEffect, useRef, useState } from "react";
import type { PipelineStage, ProjectConfig, Scene, StyleSheet } from "../api/types";
import { updateStyleSheet } from "../api/client";

interface Props {
  config: ProjectConfig;
  scenes: Scene[];
  stage: PipelineStage;
  onClose: () => void;
  onConfigUpdate: (config: ProjectConfig) => void;
}

function formatDuration(seconds: number | null): string {
  if (seconds == null) return "--:--";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

const STAGE_LABELS: Record<PipelineStage, string> = {
  upload: "Upload",
  analyze: "Analyze",
  scenes: "Scenes",
  images: "Images",
  videos: "Videos",
  upscale: "Upscale",
  assembly: "Assembly",
};

export default function ProjectHeader({ config, scenes, stage, onClose, onConfigUpdate }: Props) {
  const withImages = scenes.filter((s) => s.reference_image).length;
  const withVideo = scenes.filter(
    (s) => s.video_clip || s.sub_clips.some((sc) => sc.video_clip),
  ).length;

  const [expanded, setExpanded] = useState(false);
  const [saving, setSaving] = useState(false);
  const saveTimer = useRef<ReturnType<typeof setTimeout>>(null);

  const ss = config.style_sheet ?? { concept: "", visual_style: "", color_palette: "", aspect_ratio: "16:9", resolution: "1280x720" };

  const handleChange = useCallback(
    (field: keyof StyleSheet, value: string) => {
      const updated: StyleSheet = { ...ss, [field]: value };
      // Optimistic local update
      onConfigUpdate({ ...config, style_sheet: updated });
      // Debounced save
      if (saveTimer.current) clearTimeout(saveTimer.current);
      setSaving(true);
      saveTimer.current = setTimeout(async () => {
        try {
          await updateStyleSheet(updated);
        } catch (e) {
          console.error("Failed to save style sheet:", e);
        }
        setSaving(false);
      }, 1000);
    },
    [ss, config, onConfigUpdate],
  );

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (saveTimer.current) clearTimeout(saveTimer.current);
    };
  }, []);

  return (
    <header className="project-header">
      <div className="header-info">
        <div className="header-title-row">
          <h1>{config.name}</h1>
          <button className="close-project-btn" onClick={onClose} title="Close project">
            &times;
          </button>
        </div>
        <div className="header-stats">
          {config.song.bpm && <span className="stat">BPM {config.song.bpm}</span>}
          {config.song.keyscale && <span className="stat">{config.song.keyscale}</span>}
          <span className="stat">
            {formatDuration(config.song.duration_seconds)}
          </span>
          <span className="stat">{scenes.length} scenes</span>
          <span className="stat">
            {withImages}/{scenes.length} images
          </span>
          <span className="stat">
            {withVideo}/{scenes.length} video
          </span>
          <span className="stage-badge">{STAGE_LABELS[stage]}</span>
          <button
            className="btn btn-sm"
            onClick={() => setExpanded(!expanded)}
            style={{ marginLeft: 8 }}
          >
            {expanded ? "Hide Theme" : "Theme"}
          </button>
          {saving && <span className="save-indicator">saving...</span>}
        </div>
      </div>

      {expanded && (
        <div className="style-sheet-panel">
          <div className="style-field">
            <label>Video Concept</label>
            <textarea
              value={ss.concept}
              onChange={(e) => handleChange("concept", e.target.value)}
              placeholder="What kind of video are we making? e.g. A late-night R&B music video in a dimly lit lounge, intimate and sensual mood, single male performer..."
              rows={2}
            />
          </div>
          <div className="style-field">
            <label>Visual Style</label>
            <input
              type="text"
              value={ss.visual_style}
              onChange={(e) => handleChange("visual_style", e.target.value)}
              placeholder="e.g. anime cel-shaded, dark cinematic, retro VHS, watercolor..."
            />
          </div>
          <div className="style-field">
            <label>Color Palette</label>
            <input
              type="text"
              value={ss.color_palette}
              onChange={(e) => handleChange("color_palette", e.target.value)}
              placeholder="e.g. neon pink and cyan, muted earth tones, high-contrast B&W..."
            />
          </div>
          <div className="style-field-row">
            <div className="style-field">
              <label>Aspect Ratio</label>
              <select
                value={ss.aspect_ratio}
                onChange={(e) => handleChange("aspect_ratio", e.target.value)}
              >
                <option value="16:9">16:9</option>
                <option value="9:16">9:16</option>
                <option value="4:3">4:3</option>
                <option value="1:1">1:1</option>
              </select>
            </div>
            <div className="style-field">
              <label>Resolution</label>
              <select
                value={ss.resolution}
                onChange={(e) => handleChange("resolution", e.target.value)}
              >
                <option value="768x512">768x512</option>
                <option value="1024x576">1024x576</option>
                <option value="1280x720">1280x720</option>
                <option value="512x768">512x768 (portrait)</option>
              </select>
            </div>
          </div>
          <p className="style-hint">
            These are injected into every image and video prompt. Edit per-scene prompts to override.
          </p>
        </div>
      )}
    </header>
  );
}
