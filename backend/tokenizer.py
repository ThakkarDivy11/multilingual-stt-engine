"""
Trilingual Character Tokenizer for Automatic Speech Recognition (Gujarati • Hindi • English).
Provides unified 250-token vocabulary mapping, NFKC script normalization, CTC collapse decoding,
and zero-loss character handling across Indo-Aryan and Latin orthographies.
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Union, Optional


# Standard Indic and Latin punctuation regex for acoustic ASR normalization
PUNCTUATION_PATTERN = re.compile(r"[\.,\?!:;\"\'\(\)\[\]\-_—/\\#\$%\^&\*~`।॥«»‹›“”…„]")


def normalize_asr_text(text: str) -> str:
    """
    Standard ASR text normalizer conforming to FLEURS / Kathbath / CommonVoice conventions.
    Acoustic CTC models emit raw phonetic characters without typographical punctuation.
    Normalizing ensures accurate CER / WER evaluation and clean CTC token alignment.
    """
    if not text:
        return ""
    # 1. NFKC normalization: ensures canonical decomposition and composition of Indic matras
    norm = unicodedata.normalize("NFKC", str(text)).strip().lower()
    # 2. Strip typographical and Indic punctuation
    norm = PUNCTUATION_PATTERN.sub(" ", norm)
    # 3. Collapse multiple whitespaces to single space
    norm = " ".join(norm.split())
    return norm


class TrilingualTokenizer:
    """
    Unified Character-level Tokenizer covering Gujarati, Devanagari (Hindi), and Latin English.
    """
    def __init__(self, vocab_path: Union[str, Path]):
        vocab_path = Path(vocab_path)
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found at: {vocab_path}")

        with open(vocab_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Support both {"char_to_id": {...}} and flat {"char": id} format
        self.char_to_id: Dict[str, int] = data["char_to_id"] if "char_to_id" in data else data
        self.id_to_char: Dict[int, str] = {int(v): k for k, v in self.char_to_id.items()}

        # Special token IDs
        self.blank_id = self.char_to_id.get("<BLANK>", 0)
        self.pad_id = self.char_to_id.get("<PAD>", 1)
        self.sos_id = self.char_to_id.get("<SOS>", 2)
        self.eos_id = self.char_to_id.get("<EOS>", 3)
        self.unk_id = self.char_to_id.get("<UNK>", 4)
        self.space_id = self.char_to_id.get(" ", 5)

        self.vocab_size = len(self.char_to_id)

    def encode(self, text: str, normalize: bool = True) -> List[int]:
        """
        Encodes a string into a list of token IDs.
        """
        if normalize:
            text = normalize_asr_text(text)
        return [self.char_to_id.get(ch, self.unk_id) for ch in text]

    def decode_ctc(self, token_ids: List[int]) -> str:
        """
        Decodes a sequence of CTC emission token IDs using greedy best-path decoding:
        1. Collapse consecutive identical tokens.
        2. Remove blank tokens (<BLANK>).
        3. Drop non-printable special tokens (<PAD>, <SOS>, <EOS>, <UNK>).
        """
        prev_token = self.blank_id
        decoded_chars: List[str] = []

        for tid in token_ids:
            if tid != self.blank_id and tid != prev_token:
                char = self.id_to_char.get(tid, "")
                if not char.startswith("<"):
                    decoded_chars.append(char)
            prev_token = tid

        # Post-process: clean and format result
        result = "".join(decoded_chars)
        # Collapse multi-spaces that might have arisen
        return " ".join(result.split())

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.char_to_id)

    def __len__(self) -> int:
        return self.vocab_size

