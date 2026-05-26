#!/usr/bin/env python3
"""Compare baseline and fine-tuned WER result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare WER JSON outputs from eval_wer.py.")
    parser.add_argument("--baseline-json", default="./results/baseline_wer.json")
    parser.add_argument("--finetuned-json", default="./results/finetuned_wer.json")
    parser.add_argument("--output-json", default="./results/wer_comparison.json")
    return parser.parse_args()


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    baseline = load_payload(Path(args.baseline_json))
    finetuned = load_payload(Path(args.finetuned_json))

    baseline_wer = float(baseline["wer"])
    finetuned_wer = float(finetuned["wer"])
    delta = finetuned_wer - baseline_wer

    comparison = {
        "baseline": {
            "path": args.baseline_json,
            "wer": baseline_wer,
            "split": baseline.get("split"),
            "n_samples": baseline.get("n_samples"),
            "model_dir": baseline.get("model_dir"),
        },
        "finetuned": {
            "path": args.finetuned_json,
            "wer": finetuned_wer,
            "split": finetuned.get("split"),
            "n_samples": finetuned.get("n_samples"),
            "model_dir": finetuned.get("model_dir"),
        },
        "delta_wer": delta,
        "improved": delta < 0,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Baseline WER:  {baseline_wer:.4f}%")
    print(f"Finetuned WER: {finetuned_wer:.4f}%")
    print(f"Delta:         {delta:+.4f}% ({'improved' if delta < 0 else 'worse'})")
    print(f"Saved comparison to {output_path}")


if __name__ == "__main__":
    main()
