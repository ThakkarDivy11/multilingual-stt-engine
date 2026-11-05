"""
Production Inference Engine for Trilingual Speech-to-Text.
Loads the fine-tuned Meta MMS-300M / Wav2Vec2 checkpoint,
handles audio preprocessing, executes greedy CTC decoding, performs automatic
language script classification, and computes exact Levenshtein accuracy metrics.
"""

import os
import time
import unicodedata
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Config

from backend.tokenizer import TrilingualTokenizer, normalize_asr_text
from datasets.audio_preprocessor import load_and_preprocess_audio


def detect_language_script(text: str) -> Tuple[str, Dict[str, float]]:
    """
    Detects language script from transcribed unicode code points:
    - Gujarati (gu): U+0A80 to U+0AFF
    - Devanagari / Hindi (hi): U+0900 to U+097F
    - Latin / English (en): a-z, A-Z
    Returns detected language code ('gu', 'hi', 'en') and confidence distribution.
    """
    gu_count = 0
    hi_count = 0
    en_count = 0

    for ch in text:
        cp = ord(ch)
        if 0x0A80 <= cp <= 0x0AFF:
            gu_count += 1
        elif 0x0900 <= cp <= 0x097F:
            hi_count += 1
        elif (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A):
            en_count += 1

    total = gu_count + hi_count + en_count
    if total == 0:
        return "auto", {"gu": 0.33, "hi": 0.33, "en": 0.34}

    conf = {
        "gu": round(gu_count / total, 3),
        "hi": round(hi_count / total, 3),
        "en": round(en_count / total, 3)
    }

    if gu_count >= hi_count and gu_count >= en_count:
        return "gu", conf
    elif hi_count >= gu_count and hi_count >= en_count:
        return "hi", conf
    else:
        return "en", conf


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes exact minimum edit distance between two sequences."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def compute_accuracy_metrics(reference: str, hypothesis: str) -> Dict[str, Any]:
    """
    Calculates exact Character Error Rate (CER) and Word Error Rate (WER)
    via mathematical Levenshtein distance with zero estimations.
    """
    ref_norm = normalize_asr_text(reference)
    hyp_norm = normalize_asr_text(hypothesis)

    char_dist = levenshtein_distance(ref_norm, hyp_norm)
    total_chars = max(1, len(ref_norm))
    cer = round((char_dist / total_chars) * 100.0, 2)

    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()
    word_dist = levenshtein_distance(" ".join(ref_words), " ".join(hyp_words))
    total_words = max(1, len(ref_words))
    wer = round((word_dist / total_words) * 100.0, 2)

    return {
        "reference": reference,
        "hypothesis": hypothesis,
        "norm_reference": ref_norm,
        "norm_hypothesis": hyp_norm,
        "cer": cer,
        "wer": wer,
        "char_errors": char_dist,
        "total_chars": total_chars,
        "word_errors": word_dist,
        "total_words": total_words,
        "perfect_match": (char_dist == 0)
    }


class STTInferenceEngine:
    """
    High-performance multilingual inference engine for Wav2Vec2 / MMS-300M.
    """
    def __init__(self, checkpoint_path: Union[str, Path], vocab_path: Optional[Union[str, Path]] = None, device: Optional[str] = None):
        self.checkpoint_path = Path(checkpoint_path)
        if vocab_path is None:
            vocab_path = self.checkpoint_path / "vocab.json"
            if not vocab_path.exists():
                vocab_path = Path(__file__).resolve().parent / "vocab.json"

        self.tokenizer = TrilingualTokenizer(vocab_path)

        # Device selection: CUDA -> MPS -> CPU
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"🧠 Loading STT model from {self.checkpoint_path} onto {self.device} ...")
        self.model = Wav2Vec2ForCTC.from_pretrained(
            str(self.checkpoint_path),
            ctc_loss_reduction="mean",
            pad_token_id=self.tokenizer.blank_id,
            vocab_size=len(self.tokenizer)
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ STT Inference Engine loaded successfully ({sum(p.numel() for p in self.model.parameters()):,} parameters).")

    def transcribe(
        self,
        audio_source: Union[str, Path, bytes],
        language: str = "auto",
        reference_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribes incoming audio, detects language, and computes Levenshtein metrics if reference provided.
        """
        t_start = time.time()

        # 1. Preprocess audio
        waveform, sr, meta = load_and_preprocess_audio(audio_source)
        duration_sec = meta["duration_sec"]

        # 2. Acoustic Standardization
        mean = np.mean(waveform)
        std = np.std(waveform) + 1e-7
        norm_audio = (waveform - mean) / std

        # 3. Model Forward Pass
        input_tensor = torch.tensor(norm_audio, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    logits = self.model(input_tensor).logits
            else:
                logits = self.model(input_tensor).logits

        # 4. CTC Greedy Decoding
        pred_ids = torch.argmax(logits, dim=-1)[0].cpu().numpy().tolist()
        predicted_text = self.tokenizer.decode_ctc(pred_ids)

        # 5. Language Detection
        detected_lang, lang_conf = detect_language_script(predicted_text)
        active_lang = language if language in ["gu", "hi", "en"] else detected_lang

        inference_time_ms = round((time.time() - t_start) * 1000.0, 1)
        rtf = round((inference_time_ms / 1000.0) / max(0.001, duration_sec), 3)

        result: Dict[str, Any] = {
            "predicted_text": predicted_text,
            "language": active_lang,
            "detected_language": detected_lang,
            "language_confidence": lang_conf,
            "duration_sec": duration_sec,
            "inference_time_ms": inference_time_ms,
            "real_time_factor": rtf,
            "model_id": "divy3108/multilingual-stt-model",
            "audio_metadata": meta
        }

        # 6. Optional Reference Metric Evaluation
        if reference_text and reference_text.strip():
            result["benchmark_metrics"] = compute_accuracy_metrics(reference_text.strip(), predicted_text)

        return result
