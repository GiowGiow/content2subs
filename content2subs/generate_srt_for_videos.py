#!/usr/bin/env python3
"""
Script that scans a directory for supported media files (video/audio),
uses OpenAI Whisper to generate .srt subtitles, and optionally burns them
onto a new .mp4 for video files.

Usage (example):
    python subtitle_generator.py \
        --root /path/to/videos \
        --model small \
        --srt_only true
"""

import argparse
import logging
import tempfile
from pathlib import Path
from typing import Dict, List

import ffmpeg
from utils import is_video_file, strip_extension, write_srt
import whisper
from tqdm import tqdm

SUPPORTED_EXTENSIONS = [".mkv", ".mp4", ".mp3", ".m4u", ".wav"]
VIDEO_EXTENSIONS = {".mkv", ".mp4"}  # used when burning subtitles
PCM_CODEC = "pcm_s16le"  # 16-bit PCM
AUDIO_CHANNELS = 1  # mono
AUDIO_SAMPLE_RATE = 16_000  # 16 kHz

# Subtitle styling
OUTLINE_COLOUR_HEX = "&H40000000"
BORDER_STYLE = 3

# Logging configuration
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
LOGGER = logging.getLogger(__name__)


def extract_audio(input_files: List[Path]) -> Dict[Path, Path]:
    """
    For each file in `input_files`, uses ffmpeg to extract a mono 16kHz .wav file
    to a temporary directory. Returns a dict mapping { original_path: extracted_wav_path }.
    """
    extracted_map = {}
    for media_path in input_files:
        base_stem = strip_extension(media_path)
        wav_path = Path(tempfile.gettempdir()) / f"{base_stem}.wav"
        LOGGER.info("Extracting audio from '%s' to '%s'", media_path, wav_path)
        (
            ffmpeg.input(str(media_path))
            .output(
                str(wav_path),
                acodec=PCM_CODEC,
                ac=AUDIO_CHANNELS,
                ar=str(AUDIO_SAMPLE_RATE),
            )
            .run(quiet=True, overwrite_output=True)
        )
        extracted_map[media_path] = wav_path
    return extracted_map


def burn_subtitles_into_video(
    video_path: Path, srt_path: Path, output_dir: Path
) -> Path:
    """
    Use ffmpeg to burn the given SRT file into `video_path`,
    saving a new .mp4 in `output_dir` named <stem>_subtitled.mp4.
    """
    output_file = output_dir / f"{strip_extension(video_path)}_subtitled.mp4"
    LOGGER.info("Burning subtitles for '%s' into '%s'", video_path, output_file)

    video_input = ffmpeg.input(str(video_path))
    audio_input = video_input.audio

    (
        ffmpeg.concat(
            video_input.filter(
                "subtitles",
                str(srt_path),
                force_style=f"OutlineColour={OUTLINE_COLOUR_HEX},BorderStyle={BORDER_STYLE}",
            ),
            audio_input,
            v=1,
            a=1,
        )
        .output(str(output_file))
        .run(quiet=True, overwrite_output=True)
    )

    return output_file


def generate_subtitles(
    media_paths: List[Path], model_name: str, srt_only: bool, output_dir: Path
):
    """
    High-level function to:
      1. Load the specified Whisper model.
      2. Extract audio from each media file.
      3. Transcribe the audio -> .srt (saved in `output_dir`).
      4. Optionally burn the .srt into a new .mp4 for video files if `srt_only` is False.
    """
    if not media_paths:
        LOGGER.info("No files to process.")
        return

    # 1) Load the Whisper model
    LOGGER.info("Loading Whisper model '%s'...", model_name)
    if model_name.endswith(".en"):
        LOGGER.warning(
            "%s is an English-only model; forcing English detection.", model_name
        )
    model = whisper.load_model(model_name)

    # 2) Extract .wav audio from each file
    audio_map = extract_audio(media_paths)

    # 3) Transcribe + create SRT
    subtitles_map = {}
    for media_path, wav_path in audio_map.items():
        stem = strip_extension(media_path)
        srt_file = output_dir / f"{stem}.srt"

        LOGGER.info("Transcribing '%s' -> '%s'", media_path, srt_file)
        result = model.transcribe(str(wav_path))
        segments = result["segments"]
        write_srt(segments, srt_file)
        subtitles_map[media_path] = srt_file

    # 4) If srt_only=False, burn subtitles into a new .mp4 for each video
    if not srt_only:
        for media_path, srt_path in subtitles_map.items():
            if is_video_file(media_path):
                output_file = burn_subtitles_into_video(
                    media_path, srt_path, output_dir
                )
                LOGGER.info("Saved subtitled video to '%s'", output_file)
            else:
                LOGGER.info("Skipping burn-in for '%s' (audio-only).", media_path)
    else:
        LOGGER.info("SRT files generated only (no burn-in).")


def parse_arguments() -> argparse.Namespace:
    """
    Returns parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Automatically generate (and optionally burn) subtitles for audio/video files."
    )

    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Path to the folder containing your media files.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="small",
        help="Which Whisper model to use (e.g. tiny, small, medium, medium.en, large).",
    )
    parser.add_argument(
        "--srt_only",
        type=lambda x: x.lower() == "true",
        default=True,
        help="If 'true', only generate .srt files. If 'false', also burn them into new .mp4 for videos.",
    )

    return parser.parse_args()


def main():
    """
    Entry point for the script: parse args, gather files, check existing .srt,
    and perform transcription & optional burn-in.
    """
    args = parse_arguments()
    root_path = Path(args.root).resolve()
    model_name = args.model
    srt_only = args.srt_only

    # Collect all files with SUPPORTED_EXTENSIONS in root_path
    if not root_path.is_dir():
        LOGGER.error(
            "Provided root '%s' is not a directory or does not exist.", root_path
        )
        return

    all_files = []
    for ext in SUPPORTED_EXTENSIONS:
        all_files.extend(root_path.glob(f"*{ext}"))

    if not all_files:
        LOGGER.info(
            "No files found in '%s' with extensions %s.",
            root_path,
            SUPPORTED_EXTENSIONS,
        )
        return

    all_files = sorted(all_files)
    print(all_files)
    return
    # Filter out those that already have a matching .srt
    LOGGER.info("Checking for existing .srt files...")
    to_subtitle = []
    for media_file in tqdm(all_files, desc="Scanning files"):
        srt_candidate = root_path / f"{strip_extension(media_file)}.srt"
        if srt_candidate.exists():
            LOGGER.info(
                "Skipping '%s': subtitle '%s' already exists.",
                media_file.name,
                srt_candidate.name,
            )
        else:
            to_subtitle.append(media_file)

    if not to_subtitle:
        LOGGER.info("No new files require subtitles.")
        return

    # Generate subtitles + optionally burn
    LOGGER.info("Generating subtitles for %d file(s)...", len(to_subtitle))
    generate_subtitles(
        media_paths=to_subtitle,
        model_name=model_name,
        srt_only=srt_only,
        output_dir=root_path,
    )

    LOGGER.info("All done!")


if __name__ == "__main__":
    main()
