#!/usr/bin/env python3
"""Build a HuggingFace DatasetDict from AI-Hub studio Training labels and audio."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Audio, Dataset, DatasetDict
from sklearn.model_selection import train_test_split


DEFAULT_LABEL_DIR = (
    "/home/sypark/dataset/004.한국인_외래어_발화/"
    "01.데이터/1.Training/1.라벨링데이터_0913_add/3.스튜디오"
)
DEFAULT_AUDIO_DIR = (
    "/home/sypark/dataset/004.한국인_외래어_발화/"
    "01.데이터/1.Training/2.원천데이터_0913_add/3.스튜디오"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert AI-Hub studio JSON labels + WAV into a DatasetDict (8:1:1 by recorderId).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--label-dir", default=DEFAULT_LABEL_DIR)
    parser.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--output-dir", default="./aihub_studio_dataset")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of speakers assigned to train.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of speakers assigned to validation (remainder goes to test).",
    )
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def parse_label_json(json_path: Path) -> Optional[Dict[str, Any]]:
    try:
        with json_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        utterance = payload["발화정보"]
        recorder = payload["녹음자정보"]
        sentence = utterance.get("stt", "").strip()
        recorder_id = recorder.get("recorderId", "").strip()
        if not sentence or not recorder_id:
            return None
        return {
            "sentence": sentence,
            "recorder_id": recorder_id,
            "json_path": str(json_path),
            "speaker_dir": json_path.parent.name,
            "stem": json_path.stem,
        }
    except (KeyError, json.JSONDecodeError, OSError):
        return None


def collect_records(label_dir: Path, audio_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    records: List[Dict[str, Any]] = []
    stats = Counter()

    for json_path in sorted(label_dir.rglob("*.json")):
        stats["json_total"] += 1
        parsed = parse_label_json(json_path)
        if parsed is None:
            stats["json_invalid"] += 1
            continue

        wav_path = audio_dir / parsed["speaker_dir"] / f"{parsed['stem']}.wav"
        if not wav_path.is_file():
            stats["missing_audio"] += 1
            continue

        records.append(
            {
                "audio": str(wav_path),
                "sentence": parsed["sentence"],
                "language": "ko",
                "recorder_id": parsed["recorder_id"],
                "json_path": parsed["json_path"],
                "speaker_dir": parsed["speaker_dir"],
            }
        )
        stats["matched"] += 1

    return records, dict(stats)


def split_by_recorder(
    records: List[Dict[str, Any]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, str]]]:
    by_recorder: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_recorder[record["recorder_id"]].append(record)

    recorder_ids = sorted(by_recorder.keys())
    if not recorder_ids:
        raise ValueError("No matched label/audio pairs found.")

    holdout_ratio = 1.0 - train_ratio
    if holdout_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("Invalid split ratios. Require train_ratio + val_ratio < 1.")

    train_recorders, holdout_recorders = train_test_split(
        recorder_ids,
        test_size=holdout_ratio,
        random_state=seed,
    )
    relative_val_ratio = val_ratio / holdout_ratio
    val_recorders, test_recorders = train_test_split(
        holdout_recorders,
        test_size=1.0 - relative_val_ratio,
        random_state=seed,
    )

    split_map = {
        "train": train_recorders,
        "validation": val_recorders,
        "test": test_recorders,
    }
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    manifest: List[Dict[str, str]] = []

    for split_name, split_recorders in split_map.items():
        for recorder_id in sorted(split_recorders):
            manifest.append({"recorder_id": recorder_id, "split": split_name})
            buckets[split_name].extend(by_recorder[recorder_id])

    return buckets, manifest


def build_dataset_dict(buckets: Dict[str, List[Dict[str, Any]]]) -> DatasetDict:
    dataset_dict = {}
    for split_name, split_records in buckets.items():
        dataset = Dataset.from_list(split_records)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
        dataset_dict[split_name] = dataset
    return DatasetDict(dataset_dict)


def save_artifacts(
    output_dir: Path,
    dataset_dict: DatasetDict,
    manifest: List[Dict[str, str]],
    build_stats: Dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))

    manifest_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "splits": {
            split: {
                "num_speakers": len({row["recorder_id"] for row in manifest if row["split"] == split}),
                "num_utterances": len(dataset_dict[split]),
            }
            for split in dataset_dict.keys()
        },
        "recorders": manifest,
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "build_stats.json").write_text(
        json.dumps(build_stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if build_stats.get("missing_audio", 0) > 0:
        log_path = output_dir / "missing_audio.log"
        log_path.write_text(
            "Some JSON labels had no matching WAV under --audio-dir. "
            "Re-run dataload.py after the source download finishes.\n",
            encoding="utf-8",
        )


def main() -> None:
    args = parse_args()
    label_dir = resolve_path(args.label_dir)
    audio_dir = resolve_path(args.audio_dir)
    output_dir = resolve_path(args.output_dir)

    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    print(f"Scanning labels: {label_dir}")
    print(f"Matching audio: {audio_dir}")
    records, scan_stats = collect_records(label_dir, audio_dir)
    print(f"Scan stats: {scan_stats}")

    buckets, manifest = split_by_recorder(
        records,
        seed=args.split_seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    dataset_dict = build_dataset_dict(buckets)

    build_stats = {
        **scan_stats,
        "split_seed": args.split_seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "num_recorders": len({record["recorder_id"] for record in records}),
        "split_utterances": {split: len(dataset_dict[split]) for split in dataset_dict},
        "split_speakers": {
            split: len({row["recorder_id"] for row in manifest if row["split"] == split})
            for split in dataset_dict
        },
    }

    save_artifacts(output_dir, dataset_dict, manifest, build_stats)
    print(dataset_dict)
    print(f"Saved dataset to {output_dir}")
    print(f"Build stats: {build_stats}")


if __name__ == "__main__":
    main()
