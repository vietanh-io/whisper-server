"""
Media handling service -- ffmpeg transcoding, workspace management, and URL
normalisation.

Every transcription job gets an isolated **workspace** consisting of:
- ``job_dir``  (under ``output_dir``) -- where final transcript files are written.
- ``input_dir`` (under ``temp_dir``)  -- where raw/uploaded media is stored
  temporarily and cleaned up after the job finishes.

Audio is always transcoded to 16 kHz mono WAV before being passed to
faster-whisper, regardless of the original format.
"""

import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests
from fastapi import UploadFile


@dataclass
class JobWorkspace:
    """Paths for a single transcription job's file I/O."""

    job_id: str
    job_dir: Path     # permanent output directory
    input_dir: Path   # temporary input directory (deleted after job)
    input_path: Path  # suggested path for the raw input file


class MediaService:
    """Stateless helpers for file I/O, ffmpeg operations, and URL handling."""

    def __init__(self, output_dir: Path, temp_dir: Path) -> None:
        self.output_dir = output_dir
        self.temp_dir = temp_dir

    def ensure_runtime_dirs(self) -> None:
        """Create output and temp directories if they don't exist yet."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.resolve()
        self.temp_dir.resolve()

    def ffmpeg_available(self) -> bool:
        """Return True if ``ffmpeg`` is found in PATH and runs successfully."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                check=False,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    # ── Workspace lifecycle ─────────────────────────────────────────────

    def create_workspace(self, source_filename: str | None) -> JobWorkspace:
        """Allocate a fresh UUID-based workspace for a new job."""
        job_id = str(uuid.uuid4())
        job_dir = self.output_dir / job_id
        input_dir = self.temp_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        input_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(source_filename or "").suffix or ".bin"
        input_path = (input_dir / "input_media").with_suffix(suffix)
        return JobWorkspace(job_id=job_id, job_dir=job_dir, input_dir=input_dir, input_path=input_path)

    def cleanup_workspace_input(self, workspace: JobWorkspace) -> None:
        """Remove the temporary input directory (job_dir with outputs is kept)."""
        shutil.rmtree(workspace.input_dir, ignore_errors=True)

    # ── File operations ─────────────────────────────────────────────────

    def save_uploaded_file(self, file: UploadFile, input_path: Path) -> None:
        """Stream an uploaded file to disk."""
        with input_path.open("wb") as file_handle:
            shutil.copyfileobj(file.file, file_handle)

    def download_media(self, url: str, target_path: Path, timeout_seconds: int) -> None:
        """Download a remote file via HTTP(S) in streaming mode."""
        with requests.get(url, stream=True, timeout=timeout_seconds) as response:
            response.raise_for_status()
            with target_path.open("wb") as file_handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file_handle.write(chunk)

    # ── URL normalisation ───────────────────────────────────────────────

    def normalize_remote_url(self, url: str) -> str:
        """Resolve a remote URL into a direct media stream URL.

        YouTube links are resolved via yt-dlp to extract the best audio
        stream URL that ffmpeg can consume directly.  All other URLs are
        returned as-is.
        """
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        if "youtube.com" in host or "youtu.be" in host:
            try:
                import yt_dlp
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "YouTube URL detected but yt-dlp is not installed. Install dependency `yt-dlp`."
                ) from exc

            with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True, "format": "bestaudio/best"}) as ydl:
                info = ydl.extract_info(url, download=False)
                direct = info.get("url")
                if not direct:
                    raise RuntimeError("Cannot resolve a stream URL from YouTube link.")
                return str(direct)
        return url

    # ── ffmpeg operations ───────────────────────────────────────────────

    def transcode_to_wav(self, source: str, output_path: Path) -> None:
        """Transcode any media source to 16 kHz mono WAV via ffmpeg.

        *source* can be a local file path or a remote URL -- ffmpeg
        handles both transparently.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            source,
            "-vn",       # discard video
            "-ac",
            "1",         # mono
            "-ar",
            "16000",     # 16 kHz sample rate (Whisper's expected input)
            "-f",
            "wav",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise RuntimeError(stderr or "ffmpeg transcode failed.")

    def probe_duration_seconds(self, source_path: Path) -> float:
        """Use ffprobe to get the duration of an audio file in seconds.

        Returns 0.0 on any failure (missing ffprobe, corrupt file, etc.).
        """
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(source_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return 0.0
        try:
            return float((result.stdout or "0").strip())
        except ValueError:
            return 0.0

    def split_audio_chunks(self, source_path: Path, chunk_seconds: int, chunk_dir: Path) -> list[Path]:
        """Split a WAV file into fixed-length chunks for batch processing.

        Returns a sorted list of chunk file paths.
        """
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_pattern = str(chunk_dir / "chunk_%04d.wav")
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source_path),
            "-f",
            "segment",
            "-segment_time",
            str(chunk_seconds),
            "-c",
            "copy",
            chunk_pattern,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise RuntimeError(stderr or "ffmpeg chunking failed.")
        return sorted(chunk_dir.glob("chunk_*.wav"))

    # ── Output file helpers ─────────────────────────────────────────────

    def build_output_links(self, workspace: JobWorkspace, output_files: dict[str, Path]) -> dict[str, str]:
        """Map output file extensions to their download URL paths."""
        return {
            ext: f"/outputs/{workspace.job_id}/{path.name}"
            for ext, path in output_files.items()
        }

    def build_output_links_by_job_dir(self, job_dir: Path, output_files: dict[str, Path]) -> dict[str, str]:
        """Same as build_output_links but takes a job_dir Path directly."""
        return {
            ext: f"/outputs/{job_dir.name}/{path.name}"
            for ext, path in output_files.items()
        }

    def resolve_output_file(self, job_id: str, filename: str) -> Path:
        """Resolve and validate an output file path for download.

        Raises FileNotFoundError if the path escapes the output directory
        or the file does not exist.
        """
        candidate = (self.output_dir / job_id / filename).resolve()
        if self.output_dir.resolve() not in candidate.parents:
            raise FileNotFoundError("Output file path is invalid.")
        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError("Output file not found.")
        return candidate
