"""
Dataset Preparation, Ingestion, and Split Generation for Trilingual STT.
Downloads/loads metadata manifests, computes linguistic audit statistics,
creates stratified Train (80%) / Val (10%) / strictly held-out Test (10%) partitions,
and generates the authoritative Dataset Audit Report.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np


BASE_URL = "https://raw.githubusercontent.com/dneweb/stt/main/complete_stt_dataset_23hrs/data/metadata"
TRAIN_URL = f"{BASE_URL}/combined_train.csv"
VAL_URL = f"{BASE_URL}/combined_val.csv"

DATA_DIR = Path(__file__).resolve().parent
METADATA_DIR = DATA_DIR / "metadata"
METADATA_DIR.mkdir(parents=True, exist_ok=True)


def download_or_load_manifests() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Downloads or reads cached combined_train.csv and combined_val.csv manifests.
    """
    train_cache = METADATA_DIR / "combined_train_raw.csv"
    val_cache = METADATA_DIR / "combined_val_raw.csv"

    if not train_cache.exists():
        print(f"📥 Fetching training manifest from {TRAIN_URL} ...")
        df_train = pd.read_csv(TRAIN_URL)
        df_train.to_csv(train_cache, index=False)
    else:
        print(f"✓ Loaded cached training manifest from {train_cache}")
        df_train = pd.read_csv(train_cache)

    if not val_cache.exists():
        print(f"📥 Fetching validation manifest from {VAL_URL} ...")
        df_val = pd.read_csv(VAL_URL)
        df_val.to_csv(val_cache, index=False)
    else:
        print(f"✓ Loaded cached validation manifest from {val_cache}")
        df_val = pd.read_csv(val_cache)

    return df_train, df_val


def audit_corpus(df: pd.DataFrame, split_name: str = "Corpus") -> Dict[str, Any]:
    """
    Computes rigorous audio and linguistic metrics across languages.
    """
    total_samples = len(df)
    total_duration_sec = float(df["duration"].sum()) if "duration" in df.columns else 0.0
    total_hours = total_duration_sec / 3600.0

    missing_transcripts = int(df["transcript"].isna().sum()) + int((df["transcript"].astype(str).str.strip() == "").sum())
    unique_speakers = int(df["speaker"].nunique()) if "speaker" in df.columns else 0

    lang_stats = {}
    for lang in ["gu", "hi", "en"]:
        sub = df[df["language"] == lang]
        sub_samples = len(sub)
        sub_sec = float(sub["duration"].sum()) if "duration" in sub.columns else 0.0
        sub_hrs = sub_sec / 3600.0
        sub_spk = int(sub["speaker"].nunique()) if "speaker" in sub.columns else 0
        lang_stats[lang] = {
            "samples": sub_samples,
            "duration_hours": round(sub_hrs, 2),
            "percentage_of_corpus": round((sub_samples / max(1, total_samples)) * 100.0, 1),
            "speakers": sub_spk
        }

    return {
        "split": split_name,
        "total_samples": total_samples,
        "total_hours": round(total_hours, 2),
        "unique_speakers": unique_speakers,
        "missing_transcripts": missing_transcripts,
        "language_distribution": lang_stats
    }


def create_stratified_partitions(df_train_raw: pd.DataFrame, df_val_raw: pd.DataFrame):
    """
    Creates clean, deterministic Train, Validation, and Independent Test splits.
    Ensures an independent held-out test split (10% of total corpus per language)
    that is strictly withheld from model optimization.
    """
    # Combine all verified rows
    df_combined = pd.concat([df_train_raw, df_val_raw], ignore_index=True)
    # Deduplicate by audio_path or filename
    df_combined["filename"] = df_combined["audio_path"].apply(lambda p: os.path.basename(str(p)))
    df_combined = df_combined.drop_duplicates(subset=["filename"]).reset_index(drop=True)

    # Clean transcripts
    df_combined["transcript"] = df_combined["transcript"].fillna("").astype(str).str.strip()
    df_combined = df_combined[df_combined["transcript"] != ""].reset_index(drop=True)

    train_splits = []
    val_splits = []
    test_splits = []

    # Stratify by language with a fixed deterministic seed
    np.random.seed(42)
    for lang in ["gu", "hi", "en"]:
        sub = df_combined[df_combined["language"] == lang].copy()
        # Shuffle deterministically
        indices = np.random.permutation(len(sub))
        sub = sub.iloc[indices].reset_index(drop=True)

        n = len(sub)
        n_test = max(100, int(0.10 * n))
        n_val = max(100, int(0.10 * n))
        n_train = n - n_test - n_val

        test_part = sub.iloc[:n_test]
        val_part = sub.iloc[n_test : n_test + n_val]
        train_part = sub.iloc[n_test + n_val :]

        test_splits.append(test_part)
        val_splits.append(val_part)
        train_splits.append(train_part)

    final_train = pd.concat(train_splits, ignore_index=True)
    final_val = pd.concat(val_splits, ignore_index=True)
    final_test = pd.concat(test_splits, ignore_index=True)

    # Save CSVs
    final_train.to_csv(METADATA_DIR / "train.csv", index=False)
    final_val.to_csv(METADATA_DIR / "val.csv", index=False)
    final_test.to_csv(METADATA_DIR / "test_independent.csv", index=False)

    print(f"✓ Saved Train Split:       {len(final_train):,} samples -> {METADATA_DIR / 'train.csv'}")
    print(f"✓ Saved Validation Split:  {len(final_val):,} samples -> {METADATA_DIR / 'val.csv'}")
    print(f"✓ Saved Independent Test:  {len(final_test):,} samples -> {METADATA_DIR / 'test_independent.csv'}")

    # Generate Audit Report
    audit_data = {
        "master_corpus": audit_corpus(df_combined, "Master Corpus"),
        "train_split": audit_corpus(final_train, "Train Split (80%)"),
        "val_split": audit_corpus(final_val, "Validation Split (10%)"),
        "test_split": audit_corpus(final_test, "Independent Test Split (10%)")
    }

    report_json = METADATA_DIR / "dataset_audit_report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)

    print(f"✓ Generated Dataset Audit Report at {report_json}")
    return audit_data


if __name__ == "__main__":
    df_t, df_v = download_or_load_manifests()
    create_stratified_partitions(df_t, df_v)
