""" Transcribe audio using OpenAI's Whisper model."""

from __future__ import annotations

from pathlib import Path
import math
import asyncio
import concurrent.futures
import threading
import whisper
from yt_lib.yt_types import Snippet, ProgressReporter, IntegerProgressAllocator
from yt_lib.utils.log_utils import get_logger
from yt_lib.audio.audio_types import AUDIO_SETTINGS

logger = get_logger(__name__)

_WHISPER_THREAD_LOCAL = threading.local()





def _get_thread_local_whisper_model(model_name: str) -> whisper.model.Whisper:
    """ Load/cache Whisper model in the *worker thread* (thread-local).
        Args:
            model_name: Name of the Whisper model to load (e.g., "small").
        Returns:
            The loaded Whisper model.
    """
    model = getattr(_WHISPER_THREAD_LOCAL, "model", None)
    cached_name = getattr(_WHISPER_THREAD_LOCAL, "model_name", None)
    if model is None or cached_name != model_name:
        _WHISPER_THREAD_LOCAL.model = whisper.load_model(model_name)
        _WHISPER_THREAD_LOCAL.model_name = model_name
    return _WHISPER_THREAD_LOCAL.model

def _transcribe_chunk_in_worker_thread(model_name: str, audio_segment) -> dict:
    """ Runs inside the dedicated worker thread.
        Args:
            model_name: Name of the Whisper model to use.
            audio_segment: Audio segment to transcribe.
        Returns:
            The transcription result as a dictionary.
    """
    model = _get_thread_local_whisper_model(model_name)
    return whisper.transcribe(model, audio_segment)


async def transcribe_wav_in_chunks(
    audio_path: Path,
    model_name: str | None = None,
    chunk_duration: float | None = None,
    overlap: float | None = None,
    *,
    yield_every_n_chunks: int = 1,
    progress_rptr: ProgressReporter | None = None,
) -> list[Snippet] | None:
    """ Async/transcribable variant of `transcribe_with_whisper`.
        This function yields control back to the event loop between chunks so the job
        can be cancelled by the caller.
        Args:
            audio_path: Path to the audio file.
            model_name: Whisper model name to use.
            chunk_duration: Duration (in seconds) of each chunk.
            overlap: Overlap (in seconds) between chunks.
            yield_every_n_chunks: Yield to loop every N chunks (default: 1).
            progress_rptr: Optional progress reporter callback.
        Returns:
            List of Snippets, or None if transcription fails.
    """


    model_name = model_name or AUDIO_SETTINGS.whisper_model
    chunk_duration = (
        AUDIO_SETTINGS.chunk_duration_seconds
        if chunk_duration is None
        else chunk_duration
    )
    overlap = AUDIO_SETTINGS.chunk_overlap_seconds if overlap is None else overlap

    logger.info(
        "(async) Transcribing %s with Whisper model '%s' in %.1fs chunks (overlap %.1fs)",
        audio_path,
        model_name,
        chunk_duration,
        overlap,
    )

    # Load and resample audio (this is blocking but usually fast-ish; keep it sync).
    if progress_rptr:
        await progress_rptr.set_message("Loading audio into Whisper")

    audio = whisper.load_audio(str(audio_path))

    if progress_rptr:
        await progress_rptr.increment()

    sample_rate = AUDIO_SETTINGS.sample_rate
    samples_per_chunk = int(chunk_duration * sample_rate)
    overlap_samples = int(overlap * sample_rate)

    if samples_per_chunk <= 0:
        raise ValueError("chunk_duration must produce at least one sample")

    if overlap_samples < 0:
        raise ValueError("overlap must be non-negative")

    if overlap_samples >= samples_per_chunk:
        raise ValueError("overlap must be smaller than chunk_duration")

    num_samples = audio.shape[0]
    stride_samples = samples_per_chunk - overlap_samples

    if num_samples <= samples_per_chunk:
        num_chunks = 1
    else:
        # The fist chunk starts at sample 0 and ends at samples_per_chunk.
        # Each subsequent chunk starts stride_samples after the previous chunk's start.
        num_chunks = math.ceil((num_samples - samples_per_chunk) / stride_samples) + 1

    stride = samples_per_chunk - overlap_samples
    if stride <= 0:
        stride = samples_per_chunk

    snippets: list[Snippet] = []

    if progress_rptr:
        await progress_rptr.set_message("Whisper transcription in progress")

    progress_allocater = None
    if progress_rptr:
        progress_allocater = IntegerProgressAllocator.from_progress(
            current=progress_rptr.current,
            total=progress_rptr.total,
            steps=num_chunks,
        )

    # Dedicated single worker thread: keeps model pinned to one thread and avoids
    # 'random thread' issues that can happen if the default threadpool hops threads.
    try:
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1,
                                                   thread_name_prefix="whisper") as ex:
            for start_sample in range(0, num_samples, stride):
                # Checkpoint: if the caller cancelled, this is where CancelledError lands.
                if yield_every_n_chunks > 0:
                    chunk_index = start_sample // stride
                    if chunk_index % yield_every_n_chunks == 0:
                        await asyncio.sleep(0)
                        if progress_rptr and progress_allocater:
                            delta = progress_allocater.delta_for_completed_steps(chunk_index + 1)

                            if delta > 0:
                                await progress_rptr.set_message(
                                        f"Transcribing chunk {chunk_index} of {num_chunks}"
                                    )
                                await progress_rptr.increment(delta)

                end_sample = min(start_sample + samples_per_chunk, num_samples)
                audio_segment = audio[start_sample:end_sample]

                if audio_segment.size == 0:
                    break

                start_time = start_sample / sample_rate
                duration = (min(end_sample, num_samples) / sample_rate) - start_time

                logger.debug(
                    "(async) Chunk %d: samples [%d:%d] -> time [%.2f, %.2f]s",
                    chunk_index,
                    start_sample,
                    end_sample,
                    start_time,
                    duration,
                )

                try:
                    chunk = await loop.run_in_executor(
                        ex, _transcribe_chunk_in_worker_thread, model_name, audio_segment
                    )
                except asyncio.CancelledError:
                    # Best effort cleanup; note: a chunk already running in the worker
                    # thread will still run to completion, but we stop awaiting more work.
                    logger.info("Transcription cancelled during chunk %d.", chunk_index)
                    raise
                if text := chunk.get("text", "").strip():
                    snippet: Snippet = {
                        "text": text,
                        "start": start_time,
                        "duration": duration,
                    }
                    snippets.append(snippet)

                if end_sample >= num_samples:
                    if progress_rptr and progress_allocater:
                        await progress_rptr.set_message("Whisper transcription completed")
                        delta = progress_allocater.final_delta()
                        if delta > 0:
                            await progress_rptr.increment(delta)
                    break
        return snippets
    finally:
        audio_path.unlink(missing_ok=True)
