import shutil
import uuid
import logging
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import settings
from app.schemas import TranscribeFormInput, TranscribeRequest, TranscribeResponse
from app.transcription import WhisperService, download_media, ffmpeg_available

app = FastAPI(title=settings.app_name)
whisper_service = WhisperService()
ffmpeg_ok = False
logger = logging.getLogger(__name__)


@app.on_event("startup")
def startup() -> None:
    global ffmpeg_ok
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_ok = ffmpeg_available()


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok" if ffmpeg_ok else "degraded",
        "service": settings.app_name,
        "env": settings.app_env,
        "ffmpeg": "available" if ffmpeg_ok else "missing",
    }


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    file: UploadFile | None = File(default=None),
    media_url: str | None = Form(default=None),
    language: str | None = Form(default=None),
    task: str = Form(default=settings.default_task),
    output_formats: str = Form(default="txt,srt"),
    vad_filter: bool = Form(default=settings.default_vad_filter),
    vad_threshold: float = Form(default=settings.default_vad_threshold),
    min_silence_duration_ms: int = Form(default=settings.default_min_silence_duration_ms),
    speech_pad_ms: int = Form(default=settings.default_speech_pad_ms),
) -> TranscribeResponse:
    if not ffmpeg_ok:
        raise HTTPException(
            status_code=503,
            detail="FFmpeg is required for transcription but was not found in PATH.",
        )
    if file is None and not media_url:
        raise HTTPException(status_code=400, detail="Provide either file or media_url.")

    form_input = TranscribeFormInput(
        media_url=media_url,
        language=language,
        task=task,
        output_formats=output_formats,
        vad_filter=vad_filter,
        vad_threshold=vad_threshold,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
    )
    request_data: TranscribeRequest = form_input.to_request()

    job_id = str(uuid.uuid4())
    job_dir = settings.output_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_dir = settings.temp_dir / job_id
    input_dir.mkdir(parents=True, exist_ok=True)

    input_path = input_dir / "input_media"
    if file is not None:
        suffix = Path(file.filename or "").suffix
        input_path = input_path.with_suffix(suffix or ".bin")
        with input_path.open("wb") as file_handle:
            shutil.copyfileobj(file.file, file_handle)
    else:
        input_path = input_path.with_suffix(".bin")
        try:
            download_media(str(request_data.media_url), input_path, settings.request_timeout_seconds)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Cannot download media_url: {exc}") from exc

    try:
        result = whisper_service.transcribe(input_path=input_path, request_data=request_data, job_dir=job_dir)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Transcription failed for job_id=%s: %s", job_id, exc)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc
    finally:
        shutil.rmtree(input_dir, ignore_errors=True)

    file_map = {
        ext: f"/outputs/{job_id}/{path.name}"
        for ext, path in result.output_files.items()
    }
    return TranscribeResponse(
        job_id=job_id,
        model=settings.whisper_model,
        language=result.language,
        duration=result.duration,
        text=result.text,
        segments=result.segments,
        output_files=file_map,
    )

@app.get("/outputs/{job_id}/{filename}")
def get_output(job_id: str, filename: str) -> FileResponse:
    candidate = settings.output_dir / job_id / filename
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Output file not found.")
    return FileResponse(path=str(candidate), filename=filename)

