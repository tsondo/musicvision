/** TypeScript interfaces matching the Pydantic models in models.py */

export type SceneType = "vocal" | "instrumental";
export type ApprovalStatus = "pending" | "approved" | "rejected";
export type VideoEngineType = "humo" | "hunyuan_avatar";

export interface SubClip {
  id: string;
  time_start: number;
  time_end: number;
  audio_segment: string | null;
  video_prompt: string | null;
  video_clip: string | null;
  status: ApprovalStatus;
  frame_count: number | null;
}

export interface Scene {
  id: string;
  order: number;
  time_start: number;
  time_end: number;
  type: SceneType;
  lyrics: string;
  section: string;

  // Frame-accurate fields (populated by engine_registry.plan_subclips)
  frame_start: number | null;
  frame_end: number | null;
  total_frames: number | null;
  subclip_frame_counts: number[] | null;
  generation_audio_segments: string[] | null;

  audio_segment: string | null;
  audio_segment_vocal: string | null;

  image_prompt: string | null;
  image_prompt_user_override: string | null;
  reference_image: string | null;
  image_status: ApprovalStatus;

  video_prompt: string | null;
  video_prompt_user_override: string | null;
  video_clip: string | null;
  video_status: ApprovalStatus;
  video_engine: VideoEngineType | null;
  video_seed: number | null;
  lip_sync: boolean | null;

  sub_clips: SubClip[];

  characters: string[];
  props: string[];
  settings: string[];

  notes: string;
}

export interface SongInfo {
  audio_file: string;
  lyrics_file: string;
  bpm: number | null;
  duration_seconds: number | null;
  keyscale: string;
}

export interface ProjectConfig {
  name: string;
  created: string;
  song: SongInfo;
  video_engine: VideoEngineType;
}

export interface UpdateSceneRequest {
  image_prompt_user_override?: string;
  video_prompt_user_override?: string;
  image_status?: ApprovalStatus;
  video_status?: ApprovalStatus;
  lip_sync?: boolean;
  notes?: string;
}

export type ImageModelType =
  | "z-image-turbo"
  | "z-image"
  | "flux-dev"
  | "flux-schnell";

export interface RegenerateImageRequest {
  model?: ImageModelType;
  seed?: number;
}

export type RenderMode = "preview" | "final";

export interface RegenerateVideoRequest {
  engine?: VideoEngineType;
  seed?: number;
  render_mode?: RenderMode;
}

export interface IntakeResult {
  status: string;
  scene_count: number;
}

export interface GenerateImagesRequest {
  scene_ids?: string[];
  model?: ImageModelType;
}

export interface GenerateVideosRequest {
  scene_ids?: string[];
  engine?: VideoEngineType;
  render_mode?: RenderMode;
}

export interface OomContext {
  engine: string;
  image_size: number;
  suggestion: string;
}

export interface BatchGenFailure {
  scene_id: string;
  error: string;
  error_type?: "oom" | "oom_skipped" | "error";
  oom_context?: OomContext;
}

export interface VramWarning {
  engine: string;
  estimated_gb: number;
  available_gb: number;
  message: string;
}

export interface BatchGenResult {
  status: string;
  generated: string[];
  failed: BatchGenFailure[];
  total: number;
  vram_warnings?: VramWarning[];
  rough_cut?: string;
}

export interface FilesystemEntry {
  name: string;
  path: string;
  is_dir: boolean;
}

export interface FilesystemListResult {
  entries: FilesystemEntry[];
  parent: string | null;
}

export interface ImportAudioResult {
  status: string;
  path: string;
  acestep_imported: boolean;
  bpm?: number | null;
  duration_seconds?: number | null;
  keyscale?: string | null;
  has_lyrics?: boolean;
}

export interface ImportLyricsResult {
  status: string;
  path: string;
}

export type PipelineStage = "upload" | "intake" | "images" | "videos";
