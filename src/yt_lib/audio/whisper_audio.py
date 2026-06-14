""" Transcribe audio using OpenAI's Whisper model."""

from __future__ import annotations

from pathlib import Path
import math
import asyncio
import concurrent.futures
import threading
import whisper
from yt_lib.yt_types import ProgressReporter, IntegerProgressAllocator
from yt_lib.utils.log_utils import get_logger
from yt_lib.audio.audio_types import AUDIO_SETTINGS # , AudioTranscriptChunk

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

def _transcribe_chunk_in_worker_thread(model_name: str, segment_audio) -> dict:
    """ Runs inside the dedicated worker thread.
        Args:
            model_name: Name of the Whisper model to use.
            segment_audio: Audio segment to transcribe.
        Returns:
            The transcription result as a dictionary.
    """
    model = _get_thread_local_whisper_model(model_name)
    return whisper.transcribe(model, segment_audio)


async def transcribe_wav_in_chunks(
    audio_path: Path,
    model_name: str = AUDIO_SETTINGS.whisper_model,
    chunk_duration: float = AUDIO_SETTINGS.chunk_duration_seconds,
    overlap: float = AUDIO_SETTINGS.chunk_overlap_seconds,
    *,
    yield_every_n_chunks: int = 1,
    progress_rptr: ProgressReporter | None = None,
) -> list[dict] | None:
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
            List of Whisper chunk dicts, or None if transcription fails.
    """
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

    chunks: list[dict] = []
    chunk_index = 0

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
                if yield_every_n_chunks > 0 and (chunk_index % yield_every_n_chunks) == 0:
                    await asyncio.sleep(0)
                    if progress_rptr and progress_allocater:
                        delta = progress_allocater.delta_for_completed_steps(chunk_index + 1)

                        if delta > 0:
                            await progress_rptr.set_message(
                                    f"Transcribing chunk {chunk_index} of {num_chunks}"
                                )
                            await progress_rptr.increment(delta)

                end_sample = min(start_sample + samples_per_chunk, num_samples)
                segment_audio = audio[start_sample:end_sample]

                if segment_audio.size == 0:
                    break

                start_time = start_sample / sample_rate
                end_time = min(num_samples, end_sample) / sample_rate

                logger.debug(
                    "(async) Chunk %d: samples [%d:%d] -> time [%.2f, %.2f]s",
                    chunk_index,
                    start_sample,
                    end_sample,
                    start_time,
                    end_time,
                )

                try:
                    chunk = await loop.run_in_executor(
                        ex, _transcribe_chunk_in_worker_thread, model_name, segment_audio
                    )
                except asyncio.CancelledError:
                    # Best effort cleanup; note: a chunk already running in the worker
                    # thread will still run to completion, but we stop awaiting more work.
                    logger.info("Transcription cancelled during chunk %d.", chunk_index)
                    raise

                chunks.append(chunk)
                chunk_index += 1

                if end_sample >= num_samples:
                    if progress_rptr and progress_allocater:
                        await progress_rptr.set_message("Whisper transcription completed")
                        delta = progress_allocater.final_delta()
                        if delta > 0:
                            await progress_rptr.increment(delta)
                    break
        return chunks
    finally:
        audio_path.unlink(missing_ok=True)
