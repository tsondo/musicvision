import { fileUrl } from "../api/client";

interface Props {
  path: string | null;
}

export default function AudioPlayer({ path }: Props) {
  if (!path) return <div className="audio-placeholder">No audio</div>;

  return (
    <audio controls preload="none" className="audio-player">
      <source src={fileUrl(path)} type="audio/wav" />
    </audio>
  );
}
