# content2subs

`content2subs` is a Python script that scans a directory for supported media files (video/audio), uses OpenAI Whisper to generate .srt subtitles, and optionally burns them onto a new .mp4 for video files.

A great use is to generate subtitles for meetings or lectures that were recorded without subtitles. This allows for easy searching and indexing the content.

## Requirements

- Python 3.10
- ffmpeg
- OpenAI Whisper

## Installation

1. Install Python 3.10 using `pyenv`:

    ```sh
    pyenv install 3.10
    pyenv local 3.10.13
    ```

2. Install dependencies using `poetry`:

    ```sh
    poetry env use 3.10
    poetry install
    ```

3. Install ffmpeg:

    ```sh
    sudo apt update
    sudo apt install ffmpeg
    ```

## Usage

To generate subtitles for media files in a specified directory, run the following command:

```sh
python subtitle_generator.py --root /path/to/videos --model small --srt_only true
