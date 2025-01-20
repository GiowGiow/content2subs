from pathlib import Path
import logging

from content2subs.constants import VIDEO_EXTENSIONS
LOGGER = logging.getLogger(__name__)

def format_timestamp(seconds: float, always_include_hours: bool = False) -> str:
    """
    Convert `seconds` to an SRT timestamp string, e.g. "00:00:03,210".
    """
    if seconds < 0:
        raise ValueError("Timestamp must be non-negative")

    milliseconds = round(seconds * 1000)
    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    secs = milliseconds // 1_000
    milliseconds -= secs * 1_000

    hours_marker = f"{hours:02d}:" if (always_include_hours or hours > 0) else ""
    return f"{hours_marker}{minutes:02d}:{secs:02d},{milliseconds:03d}"


def write_srt(transcript_segments, srt_file: Path):
    """
    Given a list of Whisper transcript segments (with start, end, text),
    write them to `srt_file` in standard SRT format.
    """
    LOGGER.info("Writing SRT to %s", srt_file)
    with srt_file.open("w", encoding="utf-8") as f:
        for i, segment in enumerate(transcript_segments, start=1):
            start_ts = format_timestamp(segment["start"], always_include_hours=True)
            end_ts = format_timestamp(segment["end"], always_include_hours=True)
            # Avoid messing up SRT arrow with text that contains "-->"
            text = segment["text"].strip().replace("-->", "->")

            f.write(f"{i}\n{start_ts} --> {end_ts}\n{text}\n\n")


def is_video_file(path: Path) -> bool:
    """
    Return True if the file extension suggests it's a video file
    (used to decide if we can burn subtitles).
    """
    return path.suffix.lower() in VIDEO_EXTENSIONS


def strip_extension(filename: Path) -> str:
    """
    Return the stem (basename without extension) of `filename`.
    """
    return filename.stem
