import argparse
from pathlib import Path

import requests


def _raise_with_server_detail(response: requests.Response) -> None:
    try:
        payload = response.json()
        detail = payload.get("detail", payload)
    except Exception:  # noqa: BLE001
        detail = response.text
    raise SystemExit(f"Request failed ({response.status_code}): {detail}")


def check_health(base_url: str, timeout: int) -> None:
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Cannot connect to server at {base_url}: {exc}") from exc

    if response.status_code != 200:
        _raise_with_server_detail(response)

    payload = response.json()
    if payload.get("ffmpeg") != "available":
        raise SystemExit("Server is up but FFmpeg is missing. Install FFmpeg and restart server.")


def send_file(base_url: str, file_path: Path, args: argparse.Namespace) -> dict:
    with file_path.open("rb") as handle:
        files = {"file": (file_path.name, handle)}
        data = {
            "language": args.language,
            "task": args.task,
            "output_formats": args.output_formats,
            "vad_filter": str(args.vad_filter).lower(),
            "vad_threshold": str(args.vad_threshold),
            "min_silence_duration_ms": str(args.min_silence_duration_ms),
            "speech_pad_ms": str(args.speech_pad_ms),
        }
        response = requests.post(
            f"{base_url}/transcribe",
            files=files,
            data=data,
            timeout=args.timeout,
        )
    if response.status_code >= 400:
        _raise_with_server_detail(response)
    return response.json()


def send_url(base_url: str, media_url: str, args: argparse.Namespace) -> dict:
    data = {
        "media_url": media_url,
        "language": args.language,
        "task": args.task,
        "output_formats": args.output_formats,
        "vad_filter": str(args.vad_filter).lower(),
        "vad_threshold": str(args.vad_threshold),
        "min_silence_duration_ms": str(args.min_silence_duration_ms),
        "speech_pad_ms": str(args.speech_pad_ms),
    }
    response = requests.post(f"{base_url}/transcribe", data=data, timeout=args.timeout)
    if response.status_code >= 400:
        _raise_with_server_detail(response)
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Send media to local whisper-server.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--file", type=Path)
    parser.add_argument("--media-url")
    parser.add_argument("--language", default=None)
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    parser.add_argument("--output-formats", default="txt,srt")
    parser.add_argument("--vad-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--min-silence-duration-ms", type=int, default=500)
    parser.add_argument("--speech-pad-ms", type=int, default=400)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    if not args.file and not args.media_url:
        raise SystemExit("Provide either --file or --media-url.")

    if args.file and not args.file.exists():
        raise SystemExit(f"File not found: {args.file}")

    check_health(args.base_url, args.timeout)

    if args.file:
        payload = send_file(args.base_url, args.file, args)
    else:
        payload = send_url(args.base_url, args.media_url, args)

    print(f"job_id: {payload['job_id']}")
    print(f"language: {payload.get('language')}")
    print(f"duration: {payload.get('duration')}")
    print("output_files:")
    for kind, path in payload.get("output_files", {}).items():
        print(f"  - {kind}: {args.base_url}{path}")


if __name__ == "__main__":
    main()

