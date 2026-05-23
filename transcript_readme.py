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
