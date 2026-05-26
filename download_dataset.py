#!/usr/bin/env python3
"""Download and persist the Emilia KO/JA dataset for Whisper fine-tuning.

Run this in tmux with the asr_train virtualenv so the notebook does not need to
stay open during the long download step.

Before running this script, add a gated dataset token to .env:
    HF_TOKEN=...
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


DEFAULT_OUTPUT_DIR = "emilia_ko_ja_dataset"
DEFAULT_CACHE_DIR = "hf_cache"
DEFAULT_LANGUAGES = ("KO", "JA")
DEFAULT_ENV_FILE = ".env"
DEFAULT_MAX_TRAIN = 30_000
DEFAULT_MAX_TEST = 3_000
NORMALIZED_COLUMNS = {"audio", "sentence", "language"}
RAW_EMILIA_COLUMNS = {"mp3", "json"}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download amphion/Emilia-Dataset shards and save a train/test DatasetDict.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="DatasetDict save_to_disk path")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="Hugging Face cache directory")
    parser.add_argument(
        "--languages",
        nargs="+",
        default=list(DEFAULT_LANGUAGES),
        help="Emilia language folders to download, for example KO JA",
    )
    parser.add_argument("--test-size", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Train/test split seed")
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE, help="Path to .env containing HF_TOKEN")
    parser.add_argument(
        "--max-train",
        type=positive_int,
        default=DEFAULT_MAX_TRAIN,
        help="Maximum train samples to save",
    )
    parser.add_argument(
        "--max-test",
        type=positive_int,
        default=DEFAULT_MAX_TEST,
        help="Maximum test samples to save",
    )
    parser.add_argument(
        "--clean-cache",
        action="store_true",
        help="Remove the Hugging Face cache directory before downloading",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if it exists")
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def load_huggingface_token(env_file: Path) -> str:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError(
            "python-dotenv is required to load HF_TOKEN from .env.\n"
            "Install it in the venv with: ./asr_train/bin/python -m pip install python-dotenv"
        ) from exc

    load_dotenv(dotenv_path=env_file)
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            f"HF_TOKEN is required for amphion/Emilia-Dataset.\n"
            f"Add HF_TOKEN=... to {env_file}, then rerun this script."
        )
    return token


def normalize_emilia_batch(batch: dict) -> dict:
    return {
        "audio": batch["mp3"],
        "sentence": batch["json"]["text"],
        "language": batch["json"]["language"].lower(),
    }


def normalize_emilia_dataset(dataset):
    return dataset.map(
        normalize_emilia_batch,
        remove_columns=dataset["train"].column_names,
    )


def load_raw_emilia_dataset(cache_dir: Path, env_file: Path, languages: list[str]):
    from datasets import load_dataset

    hf_token = load_huggingface_token(env_file)
    data_files = {"train": [f"Emilia/{language}/*.tar" for language in languages]}
    return load_dataset(
        "amphion/Emilia-Dataset",
        data_files=data_files,
        cache_dir=str(cache_dir / "datasets"),
        token=hf_token,
    )


def split_and_normalize_dataset(dataset, test_size: float, seed: int, max_train: int, max_test: int):
    from datasets import DatasetDict

    split_dataset = dataset["train"].train_test_split(test_size=test_size, seed=seed)
    common_voice = DatasetDict(
        {
            "train": split_dataset["train"],
            "test": split_dataset["test"],
        }
    )
    common_voice = limit_dataset_splits(common_voice, max_train, max_test)
    return normalize_emilia_dataset(common_voice)


def limit_dataset_splits(dataset, max_train: int, max_test: int):
    from datasets import DatasetDict

    return DatasetDict(
        {
            "train": dataset["train"].select(range(min(max_train, len(dataset["train"])))),
            "test": dataset["test"].select(range(min(max_test, len(dataset["test"])))),
        }
    )


def normalize_existing_dataset(
    output_dir: Path,
    cache_dir: Path,
    env_file: Path,
    languages: list[str],
    test_size: float,
    seed: int,
) -> None:
    from datasets import load_from_disk

    dataset = load_from_disk(str(output_dir))
    train_columns = set(dataset["train"].column_names)

    if train_columns == NORMALIZED_COLUMNS:
        print(f"{output_dir} is already normalized.")
        print(dataset)
        return

    if train_columns == {"audio", "sentence"}:
        print(f"{output_dir} is usable without a language column.")
        print("The notebook infers ko/ja from audio['path'], so no extra normalization map is needed.")
        print(dataset)
        return

    if RAW_EMILIA_COLUMNS.issubset(train_columns):
        print(f"{output_dir} is a raw Emilia dataset and is usable as-is.")
        print("The notebook reads mp3/json directly, so no extra normalization map is needed.")
        print(dataset)
        return

    if not RAW_EMILIA_COLUMNS.issubset(train_columns):
        raise ValueError(
            f"{output_dir} has unsupported columns: {sorted(train_columns)}. "
            "Expected raw Emilia columns with mp3/json or normalized audio/sentence columns."
        )

    tmp_dir = output_dir.with_name(f"{output_dir.name}_normalized_tmp")
    backup_dir = output_dir.with_name(f"{output_dir.name}_raw_backup")

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    print(f"Normalizing existing dataset: {output_dir}")
    normalized = normalize_emilia_dataset(dataset)
    normalized.save_to_disk(str(tmp_dir))

    if backup_dir.exists():
        shutil.rmtree(output_dir)
        print(f"Existing raw dataset backup kept at: {backup_dir}")
    else:
        output_dir.rename(backup_dir)
        print(f"Raw dataset backup: {backup_dir}")
    tmp_dir.rename(output_dir)

    print("Done.")
    print(normalized)


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    cache_dir = resolve_path(args.cache_dir)
    env_file = resolve_path(args.env_file)
    languages = [language.upper() for language in args.languages]

    if output_dir.exists():
        if not args.overwrite:
            normalize_existing_dataset(
                output_dir=output_dir,
                cache_dir=cache_dir,
                env_file=env_file,
                languages=languages,
                test_size=args.test_size,
                seed=args.seed,
            )
            return
        shutil.rmtree(output_dir)

    if args.clean_cache and cache_dir.exists():
        print(f"Removing Hugging Face cache: {cache_dir}")
        shutil.rmtree(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "datasets"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_dir / "hub"))

    print("Downloading dataset...")
    print(f"  languages : {', '.join(languages)}")
    print(f"  cache     : {cache_dir}")
    print(f"  output    : {output_dir}")
    print(f"  max train : {args.max_train}")
    print(f"  max test  : {args.max_test}")

    dataset = load_raw_emilia_dataset(cache_dir, env_file, languages)
    common_voice = split_and_normalize_dataset(
        dataset,
        args.test_size,
        args.seed,
        args.max_train,
        args.max_test,
    )

    common_voice.save_to_disk(str(output_dir))
    print("Done.")
    print(common_voice)


if __name__ == "__main__":
    main()
