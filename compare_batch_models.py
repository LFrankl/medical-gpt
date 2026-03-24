#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run batch inference for multiple models and merge outputs into one comparison file.

Example:
python compare_batch_models.py \
  --prompts_file ./data/custom/eval/prompts.txt \
  --output_dir ./outputs/predictions/exp01 \
  --model base=Qwen/Qwen2.5-1.5B-Instruct \
  --model sft=./outputs/merged/qwen2.5-1.5b-med-sft-v1-merged \
  --model dpo=./outputs/merged/qwen2.5-1.5b-med-dpo-v1-merged
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Tuple


def parse_model_spec(spec: str) -> Tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"Invalid --model value: {spec}. Expected label=path")
    label, path = spec.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"Invalid --model value: {spec}. Expected non-empty label and path")
    return label, path


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_prompts(prompts_file: str) -> List[str]:
    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {prompts_file}")
    return prompts


def load_jsonl(file_path: str) -> List[Dict]:
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(file_path: str, rows: List[Dict]) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


def build_inference_command(
    python_bin: str,
    base_model: str,
    prompts_file: str,
    output_file: str,
    max_new_tokens: int,
    temperature: float,
    eval_batch_size: int,
    repetition_penalty: float,
    system_prompt: str,
    stop_str: str,
    load_in_4bit: bool,
    load_in_8bit: bool,
) -> List[str]:
    cmd = [
        python_bin,
        "inference.py",
        "--base_model",
        base_model,
        "--data_file",
        prompts_file,
        "--output_file",
        output_file,
        "--max_new_tokens",
        str(max_new_tokens),
        "--temperature",
        str(temperature),
        "--eval_batch_size",
        str(eval_batch_size),
        "--repetition_penalty",
        str(repetition_penalty),
    ]
    if system_prompt:
        cmd.extend(["--system_prompt", system_prompt])
    if stop_str:
        cmd.extend(["--stop_str", stop_str])
    if load_in_4bit:
        cmd.append("--load_in_4bit")
    if load_in_8bit:
        cmd.append("--load_in_8bit")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch compare multiple models with the same prompts file.")
    parser.add_argument("--prompts_file", required=True, help="Text file, one prompt per line.")
    parser.add_argument("--output_dir", required=True, help="Directory to save raw outputs and merged comparison.")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec in label=path form. Repeat this argument for multiple models.",
    )
    parser.add_argument("--python_bin", default=sys.executable, help="Python interpreter used to call inference.py")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--system_prompt", default="", help="Optional system prompt passed to inference.py")
    parser.add_argument("--stop_str", default="", help="Optional stop string passed to inference.py")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    args = parser.parse_args()

    models = [parse_model_spec(item) for item in args.model]
    ensure_dir(args.output_dir)
    prompts = load_prompts(args.prompts_file)

    raw_outputs = {}
    for label, model_path in models:
        output_file = os.path.join(args.output_dir, f"{label}.jsonl")
        cmd = build_inference_command(
            python_bin=args.python_bin,
            base_model=model_path,
            prompts_file=args.prompts_file,
            output_file=output_file,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            eval_batch_size=args.eval_batch_size,
            repetition_penalty=args.repetition_penalty,
            system_prompt=args.system_prompt,
            stop_str=args.stop_str,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
        )
        print(f"[compare] running {label}: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        rows = load_jsonl(output_file)
        if len(rows) != len(prompts):
            raise ValueError(
                f"Model {label} produced {len(rows)} rows, but prompts file has {len(prompts)} prompts"
            )
        raw_outputs[label] = rows

    merged_rows = []
    for idx, prompt in enumerate(prompts):
        row = {
            "id": idx,
            "prompt": prompt,
        }
        for label, _ in models:
            row[label] = raw_outputs[label][idx]["Output"]
        merged_rows.append(row)

    merged_file = os.path.join(args.output_dir, "comparison.jsonl")
    write_jsonl(merged_file, merged_rows)
    summary_file = os.path.join(args.output_dir, "README.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("Model comparison outputs\n")
        f.write(f"prompts_file={args.prompts_file}\n")
        f.write(f"comparison_file={merged_file}\n")
        for label, path in models:
            f.write(f"{label}={path}\n")

    print(f"[compare] saved merged comparison to {merged_file}")


if __name__ == "__main__":
    main()
