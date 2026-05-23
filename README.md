# yt_lib

High-level utilities for working with YouTube data.

---

# Features

- Fetch video metadata using the Python `yt-dlp` module
- Search YouTube
- Retrieve and process transcripts
- Generate transcripts from audio
- Manage cached video information

---

# Main Modules

| Module | Purpose |
|---|---|
| `yt_audio_transcript.py` | Generate a transcript from a YouTube video or audio source |
| `yt_ids.py` | Methods and classes for working with YouTube IDs |
| `yt_search.py` | Search YouTube and retrieve video URLs and metadata |
| `yt_transcript.py` | Retrieve and process transcripts from YouTube videos |
| `yt_video_info.py` | Fetch and manage video metadata using `yt-dlp` |

---

# Utility Modules

| Module | Purpose |
|---|---|
| `api_vault.py` | Load and manage API keys from the `.env` file |
| `app_context.py` | Manage application context, directories, and locale information |
| `log_utils.py` | Utilities for creating and managing log files |

---

# Requirements

- Python 3.12+
- `uv`
- `ffmpeg` (required for audio transcription)
- `yt-dlp` (used internally by the `yt_video_info` module)
- `Deno` (required for the `yt_video_info` module.)
- `Google API key` for YouTube search functionality

> [!NOTE]
> The project uses the Python `yt-dlp` module internally which in turn relies on the `yt-dlp` command-line tool and `Deno`. The `yt_search` module also requires `ffmpeg` and a `Google API` key.'

---

# Environment Setup

Create a `.env` file in the project root directory.

Example:

```env
GOOGLE_KEY="your_google_api_key"
OPENAI_KEY="your_openai_api_key"
```

| Key | Purpose |
|---|---|
| `GOOGLE_KEY` | Used by the YouTube search tool |
| `OPENAI_KEY` | Used by OpenAI-assisted query normalization |

---

# Examples

## Example: YouTube Search

```python
from yt_lib.yt_search import yt_search

results = yt_search(
    "Center of mass frame of reference",
    max_results=3,
)

for item in results["items"]:
    print(item["title"])
    print(item["url"])
    print()
```

Example output:

```text
Solve elastic collisions using the center of mass frame trick!
https://www.youtube.com/watch?v=P9cvU6HPZ8c

DD.2.1 Position in the CM Frame
https://www.youtube.com/watch?v=-M8swpL-Ij8

Center of mass reference frame to solve a 1D inelastic collision
https://www.youtube.com/watch?v=J5WnwdRiCj0
```

---

## Example: Transcript Retrieval and Caching

The `yt_transcript` module supports transcript caching to avoid redundant API calls and improve performance.

To enable caching, provide an object that implements a `transcript_path()` method and pass it to `set_context()`.

```python
from pathlib import Path

from yt_lib.yt_transcript import set_context, yt_json


class Cache:
    def __init__(
        self,
        cache_dir: Path = Path("./cache"),
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def transcript_path(
        self,
        video_id: str,
    ) -> Path:
        return self.cache_dir / f"{video_id}.json"


cache = Cache()
set_context(cache)

result = yt_json(
    "https://www.youtube.com/watch?v=P9cvU6HPZ8c"
)

print(f"Line 1: {result[0]["text"]}")
print(f"Start: {result[0]["start"]}")
print(f"Duration: {result[0]["duration"]}")
print()
print(f"Line 2: {result[1]["text"]}")
print(f"Start: {result[1]["start"]}")
print(f"Duration: {result[1]["duration"]}")

```

Example output:

```text
Line 1: in this video we're solving a perfectly elastic 
Collision in one dimension by transforming to  
Start: 0.44
Duration: 5.32

Line 2: the center of mass reference frame so if you've 
never seen this trick before the idea is that  
Start: 5.76
Duration: 5.72
```
### Notes

- Transcript results are returned as lists of snippet dictionaries.
- Each snippet contains:
  - `text`
  - `start`
  - `duration`
- Cached transcripts are stored as JSON files.


## Example: yt_video_info.py

The `yt_video_info` module provides a convenient interface for fetching and managing video metadata using the `yt-dlp` module.

```python
from yt_lib.yt_video_info import fetch_video_info

info = fetch_video_info("https://www.youtube.com/watch?v=-M8swpL-Ij8",
                        format_selector="bestvideo+bestaudio/best",
                        include_raw=False,
                        include_formats=False,
                    )
print(info.title)
print(info.best_format.computed_resolution)
print(info.best_format.computed_fps)
print(info.duration)
```
Example output:

```text
Title: Center of mass reference frame to solve a 1D inelastic collision.  Center of mass frame collision.
Resolution: 640x360
Frames Per Second: 30.0
Duration: 270
```
---

### Notes

- `ffmpeg` must be available in the system `PATH` for audio transcription features.
- `Deno` must be installed for the `yt_video_info` module to function correctly.

# Sources:
## [yt_lib GitHub Repository](https://github.com/Mark_USN/yt_lib)
    - The subject of this readme.
    - yt_dlp is the library used by the yt_app and yt_mcp projects for fetching video metadata and managing video information. It provides a powerful and flexible interface for retrieving video details, formats, transcripts, and other relevant information from YouTube videos.

## [uv](https://docs.astral.sh/uv/)  
    - An extremely fast Python package and project manager, written in Rust.
    - May not be 'required' for the project but is recommended for managing dependencies and running the application efficiently. It provides a streamlined workflow for installing packages, running scripts, and managing virtual environments.
    - After cloning the repository `uv sync` will create a virtual environment and install the required dependencies as specified in the `pyproject.toml` file. This ensures that the project has all the necessary libraries and tools to function correctly.    

## [yt-dlp](https://github.com/yt-dlp/yt-dlp)
    - yt-dlp is a feature-rich command-line audio/video downloader.
    - In the project it is used by the `yt_video_info` module to fetch video metadata and manage video information.
    
## [ffmpeg](https://ffmpeg.org/)
    - A complete, cross-platform solution to record, convert and stream audio and video.
    - Used by the `yt_audio_transcript` module and possibly others to process audio/video data for transcription. It is a powerful tool for handling multimedia data and is essential for the audio transcription functionality in the project.
 
## [Deno](https://deno.com/)
    - Deno is the open-source JavaScript runtime for the modern web.
    - Deno is used by yt-dlp for certain operations, such as fetching video metadata and processing video information. It provides a secure and efficient runtime environment for executing JavaScript code, which is essential for the functionality of the `yt_video_info` module.

## [Google API](https://developers.google.com/youtube/registering_an_application)
    - -Instructions for registering a YouTube application and obtaining an API key.
