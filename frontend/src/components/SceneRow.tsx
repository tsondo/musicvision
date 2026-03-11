import { useCallback, useEffect, useRef, useState } from "react";
import { fileUrl } from "../api/client";
import type {
  ImageModelType,
  RegenerateImageRequest,
  RegenerateVideoRequest,
  Scene,
  SceneAudioMode,
  UpdateSceneRequest,
  VideoEngineType,
} from "../api/types";
import type { SceneGenStatus } from "../hooks/useScenes";

interface Props {
  scene: Scene;
  imageGenStatus: SceneGenStatus;
  videoGenStatus: SceneGenStatus;
  disabled?: boolean;
  onUpdate: (sceneId: string, updates: UpdateSceneRequest) => Promise<Scene>;
  onRegenImage: (sceneId: string, req: RegenerateImageRequest) => void;
  onRegenVideo: (sceneId: string, req: RegenerateVideoRequest) => void;
  onDequeueImage: (sceneId: string) => void;
  onDequeueVideo: (sceneId: string) => void;
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function SceneRow({
  scene,
  imageGenStatus,
  videoGenStatus,
  disabled,
  onUpdate,
  onRegenImage,
  onRegenVideo,
  onDequeueImage,
  onDequeueVideo,
}: Props) {
  const duration = scene.time_end - scene.time_start;
  const effectiveImagePrompt =
    scene.image_prompt_user_override || scene.image_prompt || "";
  const effectiveVideoPrompt =
    scene.video_prompt_user_override || scene.video_prompt || "";

  const [lyrics, setLyrics] = useState(scene.lyrics || "");
  const [imagePrompt, setImagePrompt] = useState(effectiveImagePrompt);
  const [videoPrompt, setVideoPrompt] = useState(effectiveVideoPrompt);
  const [imageModel, setImageModel] = useState<ImageModelType>("z-image-turbo");
  const [imageSeed, setImageSeed] = useState(-1);
  const [videoEngine, setVideoEngine] =
    useState<VideoEngineType>("humo");
  const [videoSeed, setVideoSeed] = useState(-1);
  const [regenError, setRegenError] = useState<string | null>(null);
  const [imgVersion, setImgVersion] = useState(() => Date.now());

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);

  const isGenImage = imageGenStatus === "running";
  const isImageQueued = imageGenStatus === "queued";
  const isGenVideo = videoGenStatus === "running";
  const isVideoQueued = videoGenStatus === "queued";

  const hasImage = Boolean(scene.reference_image);
  const hasVideo = Boolean(scene.video_clip) || scene.sub_clips.some((sc) => sc.video_clip);
  const hasUpscaled = Boolean(scene.upscaled_clip);

  // Resolve the first available video clip path for preview
  const videoClipPath = scene.video_clip || scene.sub_clips.find((sc) => sc.video_clip)?.video_clip || null;
  const [showVideo, setShowVideo] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const videoAudioRef = useRef<HTMLAudioElement | null>(null);

  // Bump cache-buster when assets change (e.g. first generation sets the path)
  const lastAssetsRef = useRef({ img: scene.reference_image, vid: scene.video_clip });
  if (
    lastAssetsRef.current.img !== scene.reference_image ||
    lastAssetsRef.current.vid !== scene.video_clip
  ) {
    lastAssetsRef.current = { img: scene.reference_image, vid: scene.video_clip };
    setImgVersion(Date.now());
  }

  // Sync state when scene changes externally
  const lastPromptRef = useRef({ img: effectiveImagePrompt, vid: effectiveVideoPrompt, lyr: scene.lyrics || "" });
  if (
    lastPromptRef.current.img !== effectiveImagePrompt ||
    lastPromptRef.current.vid !== effectiveVideoPrompt ||
    lastPromptRef.current.lyr !== (scene.lyrics || "")
  ) {
    lastPromptRef.current = { img: effectiveImagePrompt, vid: effectiveVideoPrompt, lyr: scene.lyrics || "" };
    if (imagePrompt !== effectiveImagePrompt) setImagePrompt(effectiveImagePrompt);
    if (videoPrompt !== effectiveVideoPrompt) setVideoPrompt(effectiveVideoPrompt);
    if (lyrics !== (scene.lyrics || "")) setLyrics(scene.lyrics || "");
  }

  // Debounced auto-save for prompts (1s after last keystroke)
  const imgDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const vidDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [imgSaved, setImgSaved] = useState(true);
  const [vidSaved, setVidSaved] = useState(true);
  const [lyrSaved, setLyrSaved] = useState(true);

  const saveImagePrompt = useCallback(() => {
    if (imgDebounceRef.current) clearTimeout(imgDebounceRef.current);
    imgDebounceRef.current = null;
    onUpdate(scene.id, { image_prompt_user_override: imagePrompt }).then(() => setImgSaved(true));
  }, [scene.id, imagePrompt, onUpdate]);

  const saveVideoPrompt = useCallback(() => {
    if (vidDebounceRef.current) clearTimeout(vidDebounceRef.current);
    vidDebounceRef.current = null;
    onUpdate(scene.id, { video_prompt_user_override: videoPrompt }).then(() => setVidSaved(true));
  }, [scene.id, videoPrompt, onUpdate]);

  const handleImagePromptChange = (val: string) => {
    setImagePrompt(val);
    setImgSaved(false);
    if (imgDebounceRef.current) clearTimeout(imgDebounceRef.current);
    imgDebounceRef.current = setTimeout(() => {
      // Save via the ref'd value at timeout time — captured in the timeout closure
      // We need to trigger via a state update, so we use a ref trick
      imgDebounceRef.current = null;
    }, 1000);
  };

  const handleVideoPromptChange = (val: string) => {
    setVideoPrompt(val);
    setVidSaved(false);
    if (vidDebounceRef.current) clearTimeout(vidDebounceRef.current);
    vidDebounceRef.current = setTimeout(() => {
      vidDebounceRef.current = null;
    }, 1000);
  };

  // Effect-based debounce save: fires when prompt changes and debounce timer expires
  const imgPromptRef = useRef(imagePrompt);
  imgPromptRef.current = imagePrompt;
  const vidPromptRef = useRef(videoPrompt);
  vidPromptRef.current = videoPrompt;

  useEffect(() => {
    if (imgSaved) return;
    const timer = setTimeout(() => {
      if (imgPromptRef.current !== effectiveImagePrompt) {
        onUpdate(scene.id, { image_prompt_user_override: imgPromptRef.current }).then(() => setImgSaved(true));
      } else {
        setImgSaved(true);
      }
    }, 1000);
    return () => clearTimeout(timer);
  }, [imagePrompt]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (vidSaved) return;
    const timer = setTimeout(() => {
      if (vidPromptRef.current !== effectiveVideoPrompt) {
        onUpdate(scene.id, { video_prompt_user_override: vidPromptRef.current }).then(() => setVidSaved(true));
      } else {
        setVidSaved(true);
      }
    }, 1000);
    return () => clearTimeout(timer);
  }, [videoPrompt]); // eslint-disable-line react-hooks/exhaustive-deps

  const lyricsRef = useRef(lyrics);
  lyricsRef.current = lyrics;
  useEffect(() => {
    if (lyrSaved) return;
    const timer = setTimeout(() => {
      if (lyricsRef.current !== (scene.lyrics || "")) {
        onUpdate(scene.id, { lyrics: lyricsRef.current }).then(() => setLyrSaved(true));
      } else {
        setLyrSaved(true);
      }
    }, 1000);
    return () => clearTimeout(timer);
  }, [lyrics]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleRegenImage = async () => {
    setRegenError(null);
    if (isImageQueued) {
      onDequeueImage(scene.id);
      return;
    }
    if (!imgSaved && imagePrompt !== effectiveImagePrompt) {
      await onUpdate(scene.id, { image_prompt_user_override: imagePrompt });
    }
    onRegenImage(scene.id, { model: imageModel, seed: imageSeed });
  };

  const handleRegenVideo = async () => {
    setRegenError(null);
    if (isVideoQueued) {
      onDequeueVideo(scene.id);
      return;
    }
    if (!vidSaved && videoPrompt !== effectiveVideoPrompt) {
      await onUpdate(scene.id, { video_prompt_user_override: videoPrompt });
    }
    onRegenVideo(scene.id, { engine: videoEngine, seed: videoSeed });
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
      className={`scene-row ${isGenImage || isGenVideo ? "generating" : isImageQueued || isVideoQueued ? "queued" : ""}`}
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
        {scene.video_width && scene.video_height ? (
          <span
            className={`res-badge${hasUpscaled ? " upscaled" : ""}`}
            title={`${scene.video_width}×${scene.video_height}${hasUpscaled ? " (upscaled)" : ""}`}
          >
            {scene.video_height}p
          </span>
        ) : hasUpscaled ? (
          <span className="res-badge upscaled" title="Upscaled">HD</span>
        ) : null}
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

      {/* Image / Video thumbnail */}
      <div
        className={`cell cell-image${hasVideo ? " has-video" : ""}`}
        onClick={() => {
          if (!hasVideo) return;
          if (showVideo) {
            // Toggle pause/play on click while video is showing
            const vid = videoRef.current;
            if (vid) {
              if (vid.paused) {
                vid.play();
                if (videoAudioRef.current) videoAudioRef.current.play();
              } else {
                vid.pause();
                if (videoAudioRef.current) videoAudioRef.current.pause();
              }
            }
          } else {
            // Start video preview
            setShowVideo(true);
            if (videoRef.current) videoRef.current.currentTime = 0;
          }
        }}
        onDoubleClick={() => {
          if (!showVideo) return;
          // Double-click stops and returns to image
          if (videoRef.current) {
            videoRef.current.pause();
            videoRef.current.currentTime = 0;
          }
          if (videoAudioRef.current) {
            videoAudioRef.current.pause();
            videoAudioRef.current.currentTime = 0;
          }
          setShowVideo(false);
        }}
      >
        {showVideo && videoClipPath ? (
          <>
            <video
              ref={videoRef}
              src={fileUrl(videoClipPath, imgVersion)}
              autoPlay
              playsInline
              onEnded={() => {
                setShowVideo(false);
                if (videoAudioRef.current) {
                  videoAudioRef.current.pause();
                  videoAudioRef.current.currentTime = 0;
                }
              }}
              onPlay={() => {
                // Sync audio playback with video
                if (scene.audio_segment && videoAudioRef.current) {
                  videoAudioRef.current.currentTime = videoRef.current?.currentTime ?? 0;
                  videoAudioRef.current.play();
                }
              }}
            />
            {scene.audio_segment && (
              <audio
                ref={videoAudioRef}
                src={fileUrl(scene.audio_segment)}
                preload="auto"
              />
            )}
            <div className="video-stop-overlay" title="Click to pause / double-click to stop">
              {videoRef.current?.paused ? "\u25B6" : "\u275A\u275A"}
            </div>
          </>
        ) : scene.reference_image ? (
          <img
            src={fileUrl(scene.reference_image, imgVersion)}
            alt={`Scene ${scene.id}`}
            loading="lazy"
          />
        ) : (
          <div className="image-placeholder">No image</div>
        )}
        {hasVideo && !showVideo && (
          <div className="video-play-overlay" title="Click to preview video">
            &#9654;
          </div>
        )}
      </div>

      {/* Image description (editable) */}
      <div className="cell cell-prompt">
        <textarea
          value={imagePrompt}
          onChange={(e) => handleImagePromptChange(e.target.value)}
          onBlur={saveImagePrompt}
          placeholder="Image description..."
          rows={4}
        />
        {!imgSaved && <span className="save-indicator">saving...</span>}
      </div>

      {/* Source (lyrics) */}
      <div className="cell cell-source">
        <textarea
          value={lyrics}
          onChange={(e) => { setLyrics(e.target.value); setLyrSaved(false); }}
          onBlur={() => {
            if (!lyrSaved && lyrics !== (scene.lyrics || "")) {
              onUpdate(scene.id, { lyrics }).then(() => setLyrSaved(true));
            }
          }}
          placeholder={scene.type === "instrumental" ? "Instrumental" : "Lyrics..."}
          rows={2}
        />
        {!lyrSaved && <span className="save-indicator">saving...</span>}
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
          disabled={isGenImage || (disabled && !isImageQueued)}
          className={`btn-regen${isImageQueued ? " queued" : ""}`}
        >
          {isGenImage
            ? "Generating..."
            : isImageQueued
              ? "Queued ✕"
              : hasImage
                ? "Regenerate"
                : "Generate"}
        </button>
      </div>

      {/* Motion description (editable) */}
      <div className="cell cell-prompt">
        <textarea
          value={videoPrompt}
          onChange={(e) => handleVideoPromptChange(e.target.value)}
          onBlur={saveVideoPrompt}
          placeholder="Motion description..."
          rows={4}
        />
        {!vidSaved && <span className="save-indicator">saving...</span>}
      </div>

      {/* Generate video controls */}
      <div className="cell cell-controls">
        <select
          value={videoEngine}
          onChange={(e) =>
            setVideoEngine(e.target.value as VideoEngineType)
          }
        >
          <option value="humo">HuMo</option>
          <option value="ltx_video">LTX-Video 2</option>
          <option value="hunyuan_avatar">HunyuanVideo Avatar</option>
        </select>
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={scene.lip_sync ?? scene.type === "vocal"}
            onChange={(e) =>
              onUpdate(scene.id, { lip_sync: e.target.checked })
            }
          />
          Lip sync
        </label>
        <div className="seed-row">
          <label>Seed</label>
          <input
            type="number"
            value={videoSeed}
            onChange={(e) => setVideoSeed(Number(e.target.value))}
          />
          {scene.video_seed != null && (
            <span className="seed-locked" title={`Last seed: ${scene.video_seed}`}>
              #{scene.video_seed}
            </span>
          )}
        </div>
        <button
          onClick={handleRegenVideo}
          disabled={isGenVideo || !scene.reference_image || (disabled && !isVideoQueued)}
          className={`btn-regen${isVideoQueued ? " queued" : ""}`}
        >
          {isGenVideo
            ? "Rendering..."
            : isVideoQueued
              ? "Queued ✕"
              : hasVideo
                ? "Regenerate"
                : "Generate"}
        </button>
        {hasVideo && (
          <label className="checkbox-label approved-toggle">
            <input
              type="checkbox"
              checked={scene.video_status === "approved"}
              onChange={(e) =>
                onUpdate(scene.id, {
                  video_status: e.target.checked ? "approved" : "pending",
                })
              }
            />
            Approved
          </label>
        )}
      </div>

      {/* Audio mixing controls — shown when scene has generated audio */}
      {scene.generated_audio && (
        <div className="cell cell-audio-mix">
          <div className="audio-mix-header">
            <span className="audio-mix-label">Audio Mix</span>
            {scene.audio_segment && scene.generated_audio && (
              <button
                className="btn-audio btn-preview-gen"
                title="Preview generated audio"
                onClick={() => {
                  const a = new Audio(fileUrl(scene.generated_audio!));
                  a.play();
                }}
              >
                Gen &#9654;
              </button>
            )}
          </div>
          <select
            value={scene.audio_mode}
            onChange={(e) =>
              onUpdate(scene.id, { audio_mode: e.target.value as SceneAudioMode })
            }
          >
            <option value="song_only">Song Only</option>
            <option value="generated_only">Generated Only</option>
            <option value="mix">Mix</option>
          </select>

          {scene.audio_mode !== "song_only" && (
            <div className="audio-mix-sliders">
              <label>
                Gen Vol
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={scene.generated_audio_volume}
                  onChange={(e) =>
                    onUpdate(scene.id, {
                      generated_audio_volume: Number(e.target.value),
                    })
                  }
                />
                <span>{Math.round(scene.generated_audio_volume * 100)}%</span>
              </label>

              {scene.audio_mode === "mix" && (
                <label>
                  Song Duck
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={scene.song_duck_volume}
                    onChange={(e) =>
                      onUpdate(scene.id, {
                        song_duck_volume: Number(e.target.value),
                      })
                    }
                  />
                  <span>{Math.round(scene.song_duck_volume * 100)}%</span>
                </label>
              )}

              <div className="fade-controls">
                <label>
                  Fade In
                  <input
                    type="number"
                    min="0"
                    max="2"
                    step="0.05"
                    value={scene.audio_fade_in}
                    onChange={(e) =>
                      onUpdate(scene.id, { audio_fade_in: Number(e.target.value) })
                    }
                  />
                  s
                </label>
                <label>
                  Fade Out
                  <input
                    type="number"
                    min="0"
                    max="2"
                    step="0.05"
                    value={scene.audio_fade_out}
                    onChange={(e) =>
                      onUpdate(scene.id, { audio_fade_out: Number(e.target.value) })
                    }
                  />
                  s
                </label>
              </div>
            </div>
          )}
        </div>
      )}

      {regenError && <div className="row-error">{regenError}</div>}
    </div>
  );
}
