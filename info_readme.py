from yt_lib.yt_video_info import fetch_video_info

info = fetch_video_info("https://www.youtube.com/watch?v=J5WnwdRiCj0",
                        format_selector="bestvideo+bestaudio/best",
                        include_raw=False,
                        include_formats=True,
                    )

print(f"Title: {info.title}")
print(f"Resolution: {info.best_format.computed_resolution}")
print(f"Frames Per Second: {info.best_format.fps}")
print(f"Duration: {info.duration}")
