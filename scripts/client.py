"""
CLI client for whisper-server.

Sends a local file or remote URL to the running server's /transcribe
endpoint and prints the transcription result.  Supports all transcription
parameters available on the server.

Usage examples:
    python scripts/client.py --file recording.mp4
    python scripts/client.py --media-url "https://example.com/audio.mp3"
    python scripts/client.py --file audio.mp3 --task translate --source-language ja
    python scripts/client.py --file audio.mp3 --task translate --source-language ja --target-language vi
    python scripts/client.py --file audio.mp3 --temperature 0.0 --no-condition-on-previous-text
"""

import argparse
from pathlib import Path

import requests


def _raise_with_server_detail(response: requests.Response) -> None:
    """Extract the ``detail`` field from a JSON error response and exit."""
    try:
        payload = response.json()
        detail = payload.get("detail", payload)
    except Exception:  # noqa: BLE001
        detail = response.text
    raise SystemExit(f"Request failed ({response.status_code}): {detail}")


def check_health(base_url: str, timeout: int) -> None:
    """Verify the server is reachable and ffmpeg is available."""
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Cannot connect to server at {base_url}: {exc}") from exc

    if response.status_code != 200:
        _raise_with_server_detail(response)

    payload = response.json()
    if payload.get("ffmpeg") != "available":
        raise SystemExit("Server is up but FFmpeg is missing. Install FFmpeg and restart server.")


def _populate_common_fields(data: dict[str, str], args: argparse.Namespace) -> None:
    """Add all transcription parameters from parsed CLI args to the form data dict.

    Optional parameters (those that can be None) are only included when
    the user explicitly provided them, so the server uses its own
    defaults for omitted fields.
    """
    if args.source_language is not None:
        data["source_language"] = args.source_language
    if args.target_language is not None:
        data["target_language"] = args.target_language
    if args.task is not None:
        data["task"] = args.task
    if args.output_formats is not None:
        data["output_formats"] = args.output_formats
    if args.model is not None:
        data["model"] = args.model
    if args.device is not None:
        data["device"] = args.device
    if args.compute_type is not None:
        data["compute_type"] = args.compute_type
    if args.beam_size is not None:
        data["beam_size"] = str(args.beam_size)
    data["word_timestamps"] = str(args.word_timestamps).lower()
    data["batch_size"] = str(args.batch_size)
    if args.batch_mode is not None:
        data["batch_mode"] = args.batch_mode
    if args.vad_mode is not None:
        data["vad_mode"] = args.vad_mode
    if args.use_batch is not None:
        data["use_batch"] = str(args.use_batch).lower()
    data["vad_filter"] = str(args.vad_filter).lower()
    data["vad_threshold"] = str(args.vad_threshold)
    data["min_silence_duration_ms"] = str(args.min_silence_duration_ms)
    data["speech_pad_ms"] = str(args.speech_pad_ms)

    # Decoding strategy
    data["temperature"] = args.temperature
    data["best_of"] = str(args.best_of)
    data["patience"] = str(args.patience)
    data["length_penalty"] = str(args.length_penalty)
    data["repetition_penalty"] = str(args.repetition_penalty)
    data["no_repeat_ngram_size"] = str(args.no_repeat_ngram_size)

    # Quality thresholds (None = let server decide)
    if args.compression_ratio_threshold is not None:
        data["compression_ratio_threshold"] = str(args.compression_ratio_threshold)
    if args.log_prob_threshold is not None:
        data["log_prob_threshold"] = str(args.log_prob_threshold)
    if args.no_speech_threshold is not None:
        data["no_speech_threshold"] = str(args.no_speech_threshold)

    # Context & prompting
    data["condition_on_previous_text"] = str(args.condition_on_previous_text).lower()
    if args.initial_prompt is not None:
        data["initial_prompt"] = args.initial_prompt
    data["prompt_reset_on_temperature"] = str(args.prompt_reset_on_temperature)
    if args.hotwords is not None:
        data["hotwords"] = args.hotwords
    if args.prefix is not None:
        data["prefix"] = args.prefix

    # Hallucination & token control
    if args.hallucination_silence_threshold is not None:
        data["hallucination_silence_threshold"] = str(args.hallucination_silence_threshold)
    data["suppress_blank"] = str(args.suppress_blank).lower()
    data["without_timestamps"] = str(args.without_timestamps).lower()
    data["max_initial_timestamp"] = str(args.max_initial_timestamp)
    if args.max_new_tokens is not None:
        data["max_new_tokens"] = str(args.max_new_tokens)


def send_file(base_url: str, file_path: Path, args: argparse.Namespace) -> dict:
    """Upload a local file to POST /transcribe and return the JSON response."""
    with file_path.open("rb") as handle:
        files = {"file": (file_path.name, handle)}
        data: dict[str, str] = {}
        _populate_common_fields(data, args)
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
    """Send a remote URL to POST /transcribe and return the JSON response."""
    data: dict[str, str] = {"media_url": media_url}
    _populate_common_fields(data, args)
    response = requests.post(f"{base_url}/transcribe", data=data, timeout=args.timeout)
    if response.status_code >= 400:
        _raise_with_server_detail(response)
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLI client for whisper-server. Transcribe local files or remote URLs.",
    )

    # ── Connection ──────────────────────────────────────────────────
    parser.add_argument("--base-url", default="http://127.0.0.1:8000",
                        help="Server base URL (default: http://127.0.0.1:8000)")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Request timeout in seconds (default: 3600)")

    # ── Input source (one required) ─────────────────────────────────
    parser.add_argument("--file", type=Path,
                        help="Local audio/video file to transcribe")
    parser.add_argument("--media-url",
                        help="Remote URL to transcribe (HTTP, S3, YouTube)")

    # ── Task & output ───────────────────────────────────────────────
    parser.add_argument("--source-language", default=None,
                        help="ISO language code (e.g. en, vi, ja). None = auto-detect")
    parser.add_argument("--target-language", default=None,
                        help="ISO target language code for translation (default: en)")
    parser.add_argument("--task", choices=["transcribe", "translate"],
                        help="transcribe or translate to target language")
    parser.add_argument("--output-formats",
                        help="Comma-separated: txt, srt, json")

    # ── Model & runtime ─────────────────────────────────────────────
    parser.add_argument("--model", help="Model name (e.g. tiny, small, large-v3)")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"])
    parser.add_argument(
        "--compute-type",
        choices=["default", "auto", "int8", "int8_float16", "int8_bfloat16",
                 "int16", "float16", "bfloat16", "float32"],
    )

    # ── Basic transcription params ──────────────────────────────────
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--word-timestamps", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-mode", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--vad-mode", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--use-batch", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--vad-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--min-silence-duration-ms", type=int, default=500)
    parser.add_argument("--speech-pad-ms", type=int, default=400)

    # ── Decoding strategy ───────────────────────────────────────────
    parser.add_argument("--temperature", default="0.0,0.2,0.4,0.6,0.8,1.0",
                        help="Comma-separated temperature fallback sequence")
    parser.add_argument("--best-of", type=int, default=5,
                        help="Candidates when sampling with temperature > 0")
    parser.add_argument("--patience", type=float, default=1.0,
                        help="Beam search patience factor")
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0,
                        help="Prevent n-gram repetitions (0 = disabled)")

    # ── Quality thresholds ──────────────────────────────────────────
    parser.add_argument("--compression-ratio-threshold", type=float, default=2.4)
    parser.add_argument("--log-prob-threshold", type=float, default=-1.0)
    parser.add_argument("--no-speech-threshold", type=float, default=0.6)

    # ── Context & prompting ─────────────────────────────────────────
    parser.add_argument("--condition-on-previous-text", action=argparse.BooleanOptionalAction, default=True,
                        help="Use previous segment as context (disable to reduce hallucination)")
    parser.add_argument("--initial-prompt", default=None,
                        help="Text prompt to guide transcription style/vocabulary")
    parser.add_argument("--prompt-reset-on-temperature", type=float, default=0.5)
    parser.add_argument("--hotwords", default=None,
                        help="Boost specific words/phrases during generation")
    parser.add_argument("--prefix", default=None,
                        help="Text prefix for the first window")

    # ── Hallucination & token control ───────────────────────────────
    parser.add_argument("--hallucination-silence-threshold", type=float, default=None,
                        help="Skip silent periods to avoid hallucination (seconds)")
    parser.add_argument("--suppress-blank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--without-timestamps", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-initial-timestamp", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Max tokens per chunk (None = no limit)")

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
    print(f"model: {payload.get('model')}")
    print(f"device: {payload.get('device')}")
    print(f"compute_type: {payload.get('compute_type')}")
    print(f"language: {payload.get('language')}")
    print(f"duration: {payload.get('duration')}")
    print(f"used_batching: {payload.get('used_batching')}")
    print(f"chunk_count: {payload.get('chunk_count')}")
    if payload.get("translated_text"):
        print(f"translated_text: {payload['translated_text'][:200]}...")
    print("output_files:")
    for kind, path in payload.get("output_files", {}).items():
        print(f"  - {kind}: {args.base_url}{path}")


if __name__ == "__main__":
    main()
