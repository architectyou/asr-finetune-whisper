#!/usr/bin/env python3
"""Evaluate Whisper WER on a saved AI-Hub studio dataset split (baseline or fine-tuned)."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import evaluate
import torch
from datasets import DatasetDict, load_from_disk
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

from train_tutorial import (
    MODEL_NAME,
    DataCollatorSpeechSeq2SeqWithPadding,
    build_prepare_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute WER for Whisper on a dataset split without training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-path", default="./aihub_studio_dataset")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--model-dir", default=None, help="Fine-tuned checkpoint directory.")
    parser.add_argument("--processor-dir", default=None, help="Processor directory (defaults to model-dir or base model).")
    parser.add_argument("--output-json", default="./results/baseline_wer.json")
    parser.add_argument("--num-proc", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--generation-max-length", type=int, default=225)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit eval samples for smoke tests.")
    parser.add_argument("--no-fp16", action="store_true")
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def load_eval_dataset(dataset_path: Path, split: str, max_samples: Optional[int]) -> DatasetDict:
    dataset_dict = load_from_disk(str(dataset_path))
    if split not in dataset_dict:
        raise KeyError(f"Split '{split}' not found in {dataset_path}")

    eval_data = dataset_dict[split]
    if max_samples is not None:
        eval_data = eval_data.select(range(min(max_samples, len(eval_data))))

    return DatasetDict({split: eval_data})


def main() -> None:
    args = parse_args()
    dataset_path = resolve_path(args.dataset_path)
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    model_id = args.model_dir or MODEL_NAME
    processor_id = args.processor_dir or args.model_dir or MODEL_NAME

    print(f"Loading dataset from {dataset_path} (split={args.split})")
    dataset_dict = load_eval_dataset(dataset_path, args.split, args.max_samples)
    eval_split = dataset_dict[args.split]
    print(dataset_dict)

    columns_to_keep = {"audio", "sentence", "language"}
    columns_to_remove = [
        column for column in eval_split.column_names if column not in columns_to_keep
    ]
    if columns_to_remove:
        dataset_dict = dataset_dict.remove_columns(columns_to_remove)

    feature_extractor = WhisperFeatureExtractor.from_pretrained(processor_id)
    tokenizer = WhisperTokenizer.from_pretrained(processor_id, task="transcribe")
    processor = WhisperProcessor.from_pretrained(processor_id, task="transcribe")

    print("Preparing dataset features...")
    dataset_dict = dataset_dict.map(
        build_prepare_dataset(feature_extractor, tokenizer),
        remove_columns=dataset_dict.column_names[args.split],
        num_proc=args.num_proc,
    )

    print(f"Loading model from {model_id}")
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
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
        output_dir=str(resolve_path("./results/eval_tmp")),
        per_device_eval_batch_size=args.eval_batch_size,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        fp16=not args.no_fp16 and torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        eval_dataset=dataset_dict[args.split],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    print("Running evaluation...")
    metrics = trainer.evaluate()
    wer = float(metrics.get("eval_wer", metrics.get("test_wer", float("nan"))))

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "split": args.split,
        "model_dir": str(model_id),
        "processor_dir": str(processor_id),
        "n_samples": len(dataset_dict[args.split]),
        "max_samples": args.max_samples,
        "wer": wer,
        "metrics": {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))},
    }

    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WER ({args.split}): {wer:.4f}%")
    print(f"Saved results to {output_json}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
