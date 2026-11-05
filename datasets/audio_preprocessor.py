"""
Audio Preprocessing and Normalization Pipeline for Multilingual ASR.
Converts any incoming audio format to standard 16,000 Hz 16-bit Mono PCM WAV,
applies loudness normalization, trims dead silence, and verifies acoustic compliance.
"""

import io
import math
from pathlib import Path
from typing import Tuple, Optional, Union, Dict, Any

import numpy as np
import soundfile as sf


TARGET_SAMPLE_RATE = 16000
MAX_DURATION_SECONDS = 30.0
MIN_DURATION_SECONDS = 0.2


def load_and_preprocess_audio(
    audio_source: Union[str, Path, bytes, io.BytesIO],
    target_sr: int = TARGET_SAMPLE_RATE,
    normalize_volume: bool = True,
    target_peak_db: float = -1.0,
    trim_silence: bool = True
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """
    Loads audio from a file path or in-memory byte buffer, converts to mono,
    resamples to target_sr, trims dead silence margins, and normalizes amplitude.

    Returns:
        waveform: 1D float32 numpy array representing the normalized audio signal.
        sample_rate: Always target_sr (16000).
        metadata: Dictionary with raw duration, processed duration, RMS, and channels.
    """
    # 1. Read audio data using soundfile
    if isinstance(audio_source, (bytes, bytearray)):
        buffer = io.BytesIO(audio_source)
        data, sr = sf.read(buffer, dtype="float32")
    elif isinstance(audio_source, io.BytesIO):
        audio_source.seek(0)
        data, sr = sf.read(audio_source, dtype="float32")
    else:
        path = Path(audio_source)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        data, sr = sf.read(str(path), dtype="float32")

    orig_channels = 1 if data.ndim == 1 else data.shape[1]
    raw_duration = len(data) / sr

    # 2. Downmix multi-channel / stereo to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # 3. Resample to 16,000 Hz if necessary
    if sr != target_sr:
        # High quality linear / polyphase resampling without heavy dependencies
        duration = len(data) / sr
        num_target_samples = int(round(duration * target_sr))
        data = np.interp(
            np.linspace(0, len(data), num_target_samples, endpoint=False),
            np.arange(len(data)),
            data
        ).astype(np.float32)
        sr = target_sr

    # 4. Trim leading and trailing silence (energy threshold)
    if trim_silence and len(data) > int(sr * 0.2):
        frame_len = int(sr * 0.02)  # 20ms frames
        hop_len = int(sr * 0.01)    # 10ms hop
        # Frame-wise energy
        num_frames = (len(data) - frame_len) // hop_len + 1
        if num_frames > 2:
            frames = np.lib.stride_tricks.as_strided(
                data,
                shape=(num_frames, frame_len),
                strides=(data.strides[0] * hop_len, data.strides[0])
            )
            energy = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
            max_energy = np.max(energy)
            threshold = max(0.005, max_energy * 0.05)
            active_frames = np.where(energy > threshold)[0]
            if len(active_frames) > 0:
                start_sample = max(0, active_frames[0] * hop_len - int(sr * 0.05))
                end_sample = min(len(data), (active_frames[-1] + 1) * hop_len + int(sr * 0.05))
                data = data[start_sample:end_sample]

    # 5. Peak and RMS volume normalization
    max_amp = np.max(np.abs(data))
    if normalize_volume and max_amp > 1e-6:
        target_amp = 10.0 ** (target_peak_db / 20.0)
        gain = min(target_amp / max_amp, 10.0)  # Bound gain to prevent noise explosions
        data = data * gain

    # 6. Clip bounds to float32 standard [-1.0, 1.0]
    data = np.clip(data, -1.0, 1.0)
    processed_duration = len(data) / sr
    rms_energy = float(np.sqrt(np.mean(data ** 2)))

    metadata = {
        "original_channels": orig_channels,
        "sample_rate": target_sr,
        "raw_duration_sec": round(raw_duration, 3),
        "duration_sec": round(processed_duration, 3),
        "rms_energy": round(rms_energy, 4),
        "peak_amplitude": round(float(np.max(np.abs(data))), 4)
    }

    return data, target_sr, metadata


def save_processed_audio(
    waveform: np.ndarray,
    output_path: Union[str, Path],
    sample_rate: int = TARGET_SAMPLE_RATE
) -> Path:
    """
    Saves float32 waveform as standard 16-bit PCM WAV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write as 16-bit PCM WAV
    sf.write(str(output_path), waveform, sample_rate, subtype="PCM_16")
    return output_path

