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
}

export interface Scene {
  id: string;
  order: number;
  time_start: number;
  time_end: number;
  type: SceneType;
  lyrics: string;

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
  notes?: string;
}
