#!/usr/bin/env python3
"""Transcribe one audio sample with a Whisper checkpoint."""

from __future__ import annotations

import argparse
import io
import random
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


LANGUAGE_PREFIX = {
    "ko": "korean",
    "ja": "japanese",
}
MODEL_WEIGHT_NAMES = {
    "pytorch_model.bin",
    "model.safetensors",
    "pytorch_model.bin.index.json",
    "model.safetensors.index.json",
}


def default_dataset_path() -> str:
    data1_dataset = Path("/data1/ls-study/emilia_ko_ja_dataset")
    if data1_dataset.exists():
        return str(data1_dataset)
    return "./emilia_ko_ja_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a quick transcription check with a fine-tuned Whisper checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", default="./whisper-small-ko-ja/checkpoint-100")
    parser.add_argument("--processor-dir", default="./whisper-small-ko-ja")
    parser.add_argument("--dataset-path", default=default_dataset_path())
    parser.add_argument(
        "--dataset-index",
        type=int,
        default=None,
        help="Specific sample index. Omit to draw a random sample from the split.",
    )
    parser.add_argument("--audio-path", default=None)
    parser.add_argument("--language", choices=LANGUAGE_PREFIX.keys(), default="ko")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible random sampling.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=225)
    parser.add_argument("--fp16", action="store_true", help="Use fp16 inference on CUDA.")
    args = parser.parse_args()

    if args.dataset_index is not None and args.audio_path is not None:
        parser.error("Use only one of --dataset-index or --audio-path.")

    return args


def resolve_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def resolve_model_id(model_id: str) -> Union[str, Path]:
    """Return a Path for local checkpoints, or the raw string for HF Hub IDs.

    A value like "openai/whisper-small" must stay as-is so from_pretrained can
    fetch it from the Hugging Face Hub. Anything that points at an existing
    file/dir, starts with "./"/"/"/"~", or contains a backslash is treated as
    a local path.
    """
    expanded = Path(model_id).expanduser()
    if expanded.exists():
        return expanded.resolve()
    if model_id.startswith((".", "/", "~")) or "\\" in model_id:
        return expanded.resolve()
    return model_id


def log(message: str) -> None:
    print(message, flush=True)


def has_model_weights(model_dir: Path) -> bool:
    if not model_dir.exists() or not model_dir.is_dir():
        return True

    return any((model_dir / name).exists() for name in MODEL_WEIGHT_NAMES)


def get_sentence(example: dict) -> Optional[str]:
    if "sentence" in example:
        return example["sentence"]
    if "json" in example and isinstance(example["json"], dict):
        return example["json"].get("text")
    return None


def get_language(example: dict, audio: dict, fallback: str) -> str:
    language = example.get("language")
    if not language and "json" in example and isinstance(example["json"], dict):
        language = example["json"].get("language")
    if language:
        return language.lower()

    audio_path = audio.get("path")
    if audio_path:
        prefix = Path(audio_path).name.split("_", 1)[0].lower()
        if prefix in LANGUAGE_PREFIX:
            return prefix

    return fallback


def load_audio_array(source: Union[str, io.BytesIO]) -> dict:
    audio_array, sampling_rate = librosa.load(source, sr=16000, mono=True)
    return {"array": audio_array, "sampling_rate": sampling_rate}


def normalize_audio(audio: dict) -> dict:
    audio_array = np.asarray(audio["array"], dtype=np.float32)
    sampling_rate = int(audio["sampling_rate"])
    if sampling_rate != 16000:
        audio_array = librosa.resample(
            y=audio_array,
            orig_sr=sampling_rate,
            target_sr=16000,
        )
        sampling_rate = 16000

    return {"array": audio_array, "sampling_rate": sampling_rate}


def decode_audio_metadata(audio: dict) -> dict:
    if "array" in audio and "sampling_rate" in audio:
        return normalize_audio(audio)

    audio_path = audio.get("path")
    if audio_path:
        return load_audio_array(audio_path)

    audio_bytes = audio.get("bytes")
    if audio_bytes:
        with tempfile.NamedTemporaryFile(suffix=".audio") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio.flush()
            return load_audio_array(temp_audio.name)

    raise ValueError(f"Unsupported audio metadata: {audio}")


def load_dataset_audio(
    dataset_path: Path,
    split: str,
    index: Optional[int],
    fallback_language: str,
    rng: Optional[random.Random] = None,
) -> Tuple[dict, Optional[str], str, int]:
    from datasets import load_from_disk

    dataset = load_from_disk(str(dataset_path))
    if split not in dataset:
        raise ValueError(f"Split '{split}' not found. Available splits: {list(dataset.keys())}")

    split_data = dataset[split]
    if index is None:
        picker = rng if rng is not None else random
        index = picker.randrange(len(split_data))

    audio_column = "audio" if "audio" in split_data.column_names else "mp3"
    example = split_data[index]
    audio_meta = example[audio_column]
    audio = decode_audio_metadata(audio_meta)
    reference = get_sentence(example)
    language = get_language(example, audio_meta, fallback_language)
    return audio, reference, language, index


def load_file_audio(audio_path: Path) -> dict:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    return load_audio_array(str(audio_path))


def transcribe(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audio: dict,
    language: str,
    device: torch.device,
    max_new_tokens: int,
    use_fp16: bool,
) -> str:
    model_inputs = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_features = model_inputs.input_features.to(device)
    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    if use_fp16:
        input_features = input_features.half()

    prompt_language = LANGUAGE_PREFIX[language]
    try:
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=prompt_language,
            task="transcribe",
            no_timestamps=True,
        )
    except TypeError:
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=prompt_language,
            task="transcribe",
        )

    generate_kwargs = {"max_new_tokens": max_new_tokens}
    if attention_mask is not None:
        generate_kwargs["attention_mask"] = attention_mask

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            **generate_kwargs,
        )

    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


def main() -> None:
    args = parse_args()
    model_dir = resolve_model_id(args.model_dir)
    processor_dir = resolve_model_id(args.processor_dir)
    device = torch.device(args.device)
    use_fp16 = args.fp16 and device.type == "cuda"

    if isinstance(model_dir, Path) and not has_model_weights(model_dir):
        raise SystemExit(
            f"No model weights found in {model_dir}. "
            "Expected model.safetensors or pytorch_model.bin. "
            "Check whether the training checkpoint was saved correctly."
        )

    if args.audio_path is not None:
        log(f"Loading audio file: {args.audio_path}")
        audio = load_file_audio(resolve_path(args.audio_path))
        reference = None
        language = args.language
        source = args.audio_path
    else:
        rng = random.Random(args.seed) if args.seed is not None else None
        if args.dataset_index is None:
            log(f"Sampling random example from {args.dataset_path}:{args.split}"
                + (f" (seed={args.seed})" if args.seed is not None else ""))
        else:
            log(f"Loading dataset sample: {args.dataset_path}:{args.split}[{args.dataset_index}]")
        audio, reference, language, chosen_index = load_dataset_audio(
            dataset_path=resolve_path(args.dataset_path),
            split=args.split,
            index=args.dataset_index,
            fallback_language=args.language,
            rng=rng,
        )
        source = f"{args.dataset_path}:{args.split}[{chosen_index}]"

    log(f"Loading processor: {processor_dir}")
    processor = WhisperProcessor.from_pretrained(str(processor_dir), task="transcribe")
    log(f"Loading model: {model_dir} on {device}")
    model = WhisperForConditionalGeneration.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float16 if use_fp16 else torch.float32,
    ).to(device)
    model.generation_config.language = None
    model.generation_config.task = None
    model.generation_config.forced_decoder_ids = None
    model.eval()

    log("Generating transcription...")
    prediction = transcribe(
        model=model,
        processor=processor,
        audio=audio,
        language=language,
        device=device,
        max_new_tokens=args.max_new_tokens,
        use_fp16=use_fp16,
    )

    print(f"source: {source}")
    print(f"language: {language}")
    if reference:
        print(f"reference: {reference}")
    print(f"prediction: {prediction}")


if __name__ == "__main__":
    main()
