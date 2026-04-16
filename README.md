# yt_lib

High-level utilities for working with YouTube data.

## Features

- Fetch video metadata via yt-dlp
- Search YouTube
- Retrieve and process transcripts
- Manage cached video information

## Example

```python
from yt_lib import fetch_info

info = fetch_info("https://youtube.com/...")
print(info.title)
```

## Background

A friend of mine was learning about AI agents and wanted to get his agent working with YouTube Transcripts.  We realized that what was needed was a MCP (Model Context Protocol) server that the agent could query and would return a transcript or transcripts on a given topic.

He wanted to get an existing transcript, if one existed, or to create a transcript from the YouTube's audio. Using Python with the FastMCP and the youtube_transcript_api.  Of course, in a process like this you learn a lot and a couple of things resulted.  First, the MCP YouTube tool needed to be divided into two pieces.  If the YouTube had transcripts it could react fairly fast.  On the other hand if the audio needed to be processed to generate the text, using the whisper library, a different process was needed.  The server needed to be able to start the job and return the job's current status when the agent queried it.  Finally, when the job was done it needed to return the results when requested by the Agent.  As a result the project became broken into two MCP servers and two transcript tools.  One simple server handling only the existing transcript case and another that could handle both that case and the audio case.

The second realization was that the Agent was not a good place to specify the YouTube video it desired.  Instead a MCP server tool should be created that would allow the Agent to search for YouTubes on some topic of interest.  The Agent could then download the transcript for YouTubes it deemed of interest.  This lead to the yt_search MCP tool which uses the googleapiclient library to search for YouTube videos and return some information about them.

Then my friend decided he wanted to have a standalone app that would be able to specify a YouTube video and get the transcript and some other information about the video.  This request resulted in two things.  First the YouTube code, which had been happily living in the MCP server project needed to become a full standalone library and second I needed a way to get information about the YouTube of interest.  I could continue using the yt_search.py code but it was kind of limited in the information it could return.  It also required an access key from Google and had use limits/costs associated with its use.  These concerns resulted in the use of the yt_dlp library to retrieve information about a YouTube.

So that is how the major elements in yt_lib came to be.  Along the way as the need for other elements was realized they were added to the library under utils.  To handle the keys needed by agents and Google api_keys came into being.  Whisper's need for ffmpeg resulted in the ffmpeg_bootstrap.py which will download and install ffmpeg if it can not find it on the system.  Consistent logging and the desire to dump some data structures out to the logs resulted in log_utils, and lastly paths.py was developed to shove cache files where I wanted them.  I turned out to be a mistake and I am planning on killing it and using an AppContext class that uses directories based on the platformdirs library.

## Still Working

At this time I still have some work to do mostly on documentation and testing for everything.  In addition the 'Long Job MCP Server' needs a good deal of clean-up and polishing.  The initial desire to allow the YouTube Audio Transcript tool to work with both the 'simple' MCP server and the 'long job' server resulted in some kluges of magnificent breadth, which I plan to remove.
