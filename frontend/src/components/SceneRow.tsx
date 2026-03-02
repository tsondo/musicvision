import { useRef, useState } from "react";
import { fileUrl } from "../api/client";
import type {
  ImageModelType,
  RegenerateImageRequest,
  RegenerateVideoRequest,
  Scene,
  UpdateSceneRequest,
  VideoEngineType,
} from "../api/types";

interface Props {
  scene: Scene;
  generating: Set<string>;
  disabled?: boolean;
  onUpdate: (sceneId: string, updates: UpdateSceneRequest) => Promise<Scene>;
  onRegenImage: (
    sceneId: string,
    req: RegenerateImageRequest,
  ) => Promise<Scene>;
  onRegenVideo: (
    sceneId: string,
    req: RegenerateVideoRequest,
  ) => Promise<Scene>;
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function SceneRow({
  scene,
  generating,
  disabled,
  onUpdate,
  onRegenImage,
  onRegenVideo,
}: Props) {
  const duration = scene.time_end - scene.time_start;
  const effectiveImagePrompt =
    scene.image_prompt_user_override || scene.image_prompt || "";
  const effectiveVideoPrompt =
    scene.video_prompt_user_override || scene.video_prompt || "";

  const [imagePrompt, setImagePrompt] = useState(effectiveImagePrompt);
  const [videoPrompt, setVideoPrompt] = useState(effectiveVideoPrompt);
  const [imageModel, setImageModel] = useState<ImageModelType>("z-image-turbo");
  const [imageSeed, setImageSeed] = useState(-1);
  const [videoEngine, setVideoEngine] =
    useState<VideoEngineType>("hunyuan_avatar");
  const [videoSeed, setVideoSeed] = useState(-1);
  const [regenError, setRegenError] = useState<string | null>(null);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);

  const isGenImage = generating.has(scene.id + ":image");
  const isGenVideo = generating.has(scene.id + ":video");

  // Sync prompt state when scene changes externally
  const lastPromptRef = useRef({ img: effectiveImagePrompt, vid: effectiveVideoPrompt });
  if (
    lastPromptRef.current.img !== effectiveImagePrompt ||
    lastPromptRef.current.vid !== effectiveVideoPrompt
  ) {
    lastPromptRef.current = { img: effectiveImagePrompt, vid: effectiveVideoPrompt };
    if (imagePrompt !== effectiveImagePrompt) setImagePrompt(effectiveImagePrompt);
    if (videoPrompt !== effectiveVideoPrompt) setVideoPrompt(effectiveVideoPrompt);
  }

  const saveImagePrompt = () => {
    if (imagePrompt !== effectiveImagePrompt) {
      onUpdate(scene.id, { image_prompt_user_override: imagePrompt });
    }
  };

  const saveVideoPrompt = () => {
    if (videoPrompt !== effectiveVideoPrompt) {
      onUpdate(scene.id, { video_prompt_user_override: videoPrompt });
    }
  };

  const handleRegenImage = async () => {
    setRegenError(null);
    // Save prompt first if changed
    if (imagePrompt !== effectiveImagePrompt) {
      await onUpdate(scene.id, { image_prompt_user_override: imagePrompt });
    }
    try {
      await onRegenImage(scene.id, { model: imageModel, seed: imageSeed });
    } catch (err) {
      setRegenError(String(err));
    }
  };

  const handleRegenVideo = async () => {
    setRegenError(null);
    if (videoPrompt !== effectiveVideoPrompt) {
      await onUpdate(scene.id, { video_prompt_user_override: videoPrompt });
    }
    try {
      await onRegenVideo(scene.id, { engine: videoEngine, seed: videoSeed });
    } catch (err) {
      setRegenError(String(err));
    }
  };

  const toggleAudio = () => {
    if (!scene.audio_segment) return;
    if (!audioRef.current) {
      audioRef.current = new Audio(fileUrl(scene.audio_segment));
      audioRef.current.onended = () => setPlaying(false);
    }
    if (playing) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      setPlaying(false);
    } else {
      audioRef.current.play();
      setPlaying(true);
    }
  };

  return (
    <div
      className={`scene-row ${isGenImage || isGenVideo ? "generating" : ""}`}
    >
      {/* Info cell */}
      <div className="cell cell-info">
        <span className="scene-order">
          {String(scene.order).padStart(2, "0")}
        </span>
        <span className="scene-time">{formatTime(scene.time_start)}</span>
        <span className="scene-duration">{duration.toFixed(1)}s</span>
        {scene.sub_clips.length > 0 && (
          <span className="sub-clip-count">
            {scene.sub_clips.length} clips
          </span>
        )}
        {scene.audio_segment && (
          <button
            className="btn-audio"
            onClick={toggleAudio}
            title={playing ? "Stop" : "Play audio"}
          >
            {playing ? "\u25A0" : "\u25B6"}
          </button>
        )}
      </div>

      {/* Image thumbnail */}
      <div className="cell cell-image">
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

      {/* Image description (editable) */}
      <div className="cell cell-prompt">
        <textarea
          value={imagePrompt}
          onChange={(e) => setImagePrompt(e.target.value)}
          onBlur={saveImagePrompt}
          placeholder="Image description..."
          rows={4}
        />
      </div>

      {/* Source (lyrics) */}
      <div className="cell cell-source">
        {scene.type === "vocal" && scene.lyrics ? (
          <div className="lyrics-text">{scene.lyrics}</div>
        ) : (
          <div className="instrumental-label">Instrumental</div>
        )}
      </div>

      {/* Generate image controls */}
      <div className="cell cell-controls">
        <select
          value={imageModel}
          onChange={(e) => setImageModel(e.target.value as ImageModelType)}
        >
          <option value="z-image-turbo">Z-Image Turbo</option>
          <option value="z-image">Z-Image</option>
          <option value="flux-dev">FLUX Dev</option>
          <option value="flux-schnell">FLUX Schnell</option>
        </select>
        <div className="seed-row">
          <label>Seed</label>
          <input
            type="number"
            value={imageSeed}
            onChange={(e) => setImageSeed(Number(e.target.value))}
          />
        </div>
        <button
          onClick={handleRegenImage}
          disabled={isGenImage || !imagePrompt || disabled}
          className="btn-regen"
        >
          {isGenImage ? "Generating..." : "Regenerate"}
        </button>
      </div>

      {/* Motion description (editable) */}
      <div className="cell cell-prompt">
        <textarea
          value={videoPrompt}
          onChange={(e) => setVideoPrompt(e.target.value)}
          onBlur={saveVideoPrompt}
          placeholder="Motion description..."
          rows={4}
        />
      </div>

      {/* Generate video controls */}
      <div className="cell cell-controls">
        <select
          value={videoEngine}
          onChange={(e) =>
            setVideoEngine(e.target.value as VideoEngineType)
          }
        >
          <option value="hunyuan_avatar">HunyuanVideo Avatar</option>
          <option value="humo">HuMo</option>
        </select>
        <div className="seed-row">
          <label>Seed</label>
          <input
            type="number"
            value={videoSeed}
            onChange={(e) => setVideoSeed(Number(e.target.value))}
          />
        </div>
        <button
          onClick={handleRegenVideo}
          disabled={isGenVideo || !videoPrompt || !scene.reference_image || disabled}
          className="btn-regen"
        >
          {isGenVideo ? "Generating..." : "Regenerate"}
        </button>
      </div>

      {regenError && <div className="row-error">{regenError}</div>}
    </div>
  );
}
