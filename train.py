#!/usr/bin/env python3
"""Train Whisper small on the local Emilia KO/JA dataset.

This is the runnable script version of fine_tune_whisper.ipynb. It accepts both
the raw Emilia schema (__key__/__url__/json/mp3) and the normalized schema
(audio/sentence/language).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import evaluate
import librosa
import numpy as np
import torch
import yaml
from datasets import load_from_disk
from dotenv import load_dotenv
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)


MODEL_NAME = "openai/whisper-small"
TARGET_SAMPLING_RATE = 16000
LANGUAGE_PREFIX = {
    "ko": "korean",
    "ja": "japanese",
}


def _load_yaml_defaults(config_path: Optional[str]) -> Dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}")
    return {key.replace("-", "_"): value for key, value in data.items()}


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    pre_args, _ = pre_parser.parse_known_args()
    yaml_defaults = _load_yaml_defaults(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper small on local ASR data (AI-Hub studio or Emilia KO/JA).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="YAML config file with default arguments.")
    parser.add_argument("--dataset-path", default="./aihub_studio_dataset")
    parser.add_argument("--output-dir", default="./whisper-small-aihub-studio")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument(
        "--language",
        choices=["all", *LANGUAGE_PREFIX.keys()],
        default="ko",
        help="Language subset to train/evaluate on.",
    )
    parser.add_argument(
        "--eval-split",
        default="validation",
        choices=["validation", "test"],
        help="Split used for evaluation during training.",
    )
    parser.add_argument("--num-proc", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--generation-max-length", type=int, default=225)
    parser.add_argument("--no-fp16", action="store_true", help="Disable fp16 training")
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (may help avoid runtime issues).",
    )
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--final-eval-json",
        default="./results/finetuned_wer.json",
        help="Path to save final WER JSON after training (set empty string to skip).",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap on train split size (useful for smoke runs).",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Optional cap on eval split size during training.",
    )

    if yaml_defaults:
        unknown = set(yaml_defaults) - {action.dest for action in parser._actions}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        parser.set_defaults(**yaml_defaults)

    return parser.parse_args()


def resolve_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def load_env(env_file: Path) -> None:
    if env_file.exists():
        load_dotenv(dotenv_path=env_file)


def get_audio(batch: dict) -> dict:
    return batch["audio"] if "audio" in batch else batch["mp3"]


def get_sentence(batch: dict) -> str:
    if "sentence" in batch:
        return batch["sentence"]
    return batch["json"]["text"]


def get_metadata_language(batch: dict) -> Optional[str]:
    language = batch.get("language")
    if not language and "json" in batch:
        language = batch["json"].get("language")
    if language:
        return language.lower()
    return None


def infer_language(batch: dict, audio: dict) -> str:
    language = get_metadata_language(batch)
    if language:
        return language

    audio_path = audio.get("path", "")
    prefix = Path(audio_path).name.split("_", 1)[0].lower()
    if prefix in LANGUAGE_PREFIX:
        return prefix

    raise ValueError(f"Could not infer language from audio path: {audio_path}")


def matches_language(batch: dict, language: str) -> bool:
    metadata_language = get_metadata_language(batch)
    if metadata_language:
        return metadata_language == language

    audio_path = get_audio(batch).get("path", "")
    return Path(audio_path).name.split("_", 1)[0].lower() == language


def build_prepare_dataset(feature_extractor: WhisperFeatureExtractor, tokenizer: WhisperTokenizer):
    def prepare_dataset(batch: dict) -> dict:
        audio = get_audio(batch)
        audio_array = np.asarray(audio["array"])
        sampling_rate = audio["sampling_rate"]
        if sampling_rate != TARGET_SAMPLING_RATE:
            audio_array = librosa.resample(
                audio_array,
                orig_sr=sampling_rate,
                target_sr=TARGET_SAMPLING_RATE,
            )
            sampling_rate = TARGET_SAMPLING_RATE

        batch["input_features"] = feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
        ).input_features[0]

        language = infer_language(batch, audio)
        tokenizer.set_prefix_tokens(
            language=LANGUAGE_PREFIX[language],
            task="transcribe",
            predict_timestamps=False,
        )
        batch["labels"] = tokenizer(get_sentence(batch)).input_ids
        return batch

    return prepare_dataset


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def main() -> None:
    args = parse_args()
    load_env(resolve_path(args.env_file))

    dataset_path = resolve_path(args.dataset_path)
    output_dir = resolve_path(args.output_dir)

    print(f"Loading dataset from {dataset_path}")
    common_voice = load_from_disk(str(dataset_path))
    print(common_voice)

    if args.language != "all":
        print(f"Filtering dataset to language: {args.language}")
        common_voice = common_voice.filter(
            lambda language: language.lower() == args.language,
            input_columns=["language"],
            num_proc=args.num_proc,
        )
        print(common_voice)

    if args.max_train_samples is not None:
        common_voice["train"] = common_voice["train"].select(
            range(min(args.max_train_samples, len(common_voice["train"])))
        )
    if args.max_eval_samples is not None:
        common_voice[args.eval_split] = common_voice[args.eval_split].select(
            range(min(args.max_eval_samples, len(common_voice[args.eval_split])))
        )

    # 학습/평가에 필요 없는 split은 제거해 전처리(map) 시간을 줄입니다.
    # (Seq2SeqTrainer는 train + eval split만 사용하지만, DatasetDict.map은 모든 split에 적용됩니다.)
    keep_splits = {"train", args.eval_split}
    for split_name in list(common_voice.keys()):
        if split_name not in keep_splits:
            common_voice.pop(split_name)

    columns_to_keep = {"audio", "sentence", "language", "mp3", "json"}
    columns_to_remove = [
        column
        for column in common_voice["train"].column_names
        if column not in columns_to_keep
    ]
    if columns_to_remove:
        common_voice = common_voice.remove_columns(columns_to_remove)

    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, task="transcribe")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, task="transcribe")

    print("Preparing dataset features...")
    common_voice = common_voice.map(
        build_prepare_dataset(feature_extractor, tokenizer),
        remove_columns=common_voice.column_names["train"],
        num_proc=args.num_proc,
    )

    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.generation_config.language = None
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        gradient_checkpointing=False,
        fp16=not args.no_fp16,
        evaluation_strategy="steps",
        per_device_eval_batch_size=args.eval_batch_size,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=args.push_to_hub,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice[args.eval_split],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    processor.save_pretrained(training_args.output_dir)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if args.final_eval_json:
        print("Running final evaluation on best model...")
        metrics = trainer.evaluate()
        wer = float(metrics.get("eval_wer", float("nan")))

        output_json = resolve_path(args.final_eval_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(dataset_path),
            "split": args.eval_split,
            "model_dir": str(output_dir),
            "language": args.language,
            "max_steps": args.max_steps,
            "n_samples": len(common_voice[args.eval_split]),
            "wer": wer,
            "metrics": {
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            },
        }
        output_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Final WER ({args.eval_split}): {wer:.4f}%")
        print(f"Saved results to {output_json}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
