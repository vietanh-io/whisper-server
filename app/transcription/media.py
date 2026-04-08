import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

import requests
from fastapi import UploadFile


@dataclass
class JobWorkspace:
    job_id: str
    job_dir: Path
    input_dir: Path
    input_path: Path


class MediaService:
    def __init__(self, output_dir: Path, temp_dir: Path) -> None:
        self.output_dir = output_dir
        self.temp_dir = temp_dir

    def ensure_runtime_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def ffmpeg_available(self) -> bool:
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

    def create_workspace(self, source_filename: str | None) -> JobWorkspace:
        job_id = str(uuid.uuid4())
        job_dir = self.output_dir / job_id
        input_dir = self.temp_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        input_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(source_filename or "").suffix or ".bin"
        input_path = (input_dir / "input_media").with_suffix(suffix)
        return JobWorkspace(job_id=job_id, job_dir=job_dir, input_dir=input_dir, input_path=input_path)

    def save_uploaded_file(self, file: UploadFile, input_path: Path) -> None:
        with input_path.open("wb") as file_handle:
            shutil.copyfileobj(file.file, file_handle)

    def download_media(self, url: str, target_path: Path, timeout_seconds: int) -> None:
        with requests.get(url, stream=True, timeout=timeout_seconds) as response:
            response.raise_for_status()
            with target_path.open("wb") as file_handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file_handle.write(chunk)

    def cleanup_workspace_input(self, workspace: JobWorkspace) -> None:
        shutil.rmtree(workspace.input_dir, ignore_errors=True)

    def build_output_links(self, workspace: JobWorkspace, output_files: dict[str, Path]) -> dict[str, str]:
        return {
            ext: f"/outputs/{workspace.job_id}/{path.name}"
            for ext, path in output_files.items()
        }

    def resolve_output_file(self, job_id: str, filename: str) -> Path:
        candidate = (self.output_dir / job_id / filename).resolve()
        if self.output_dir.resolve() not in candidate.parents:
            raise FileNotFoundError("Output file path is invalid.")
        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError("Output file not found.")
        return candidate

