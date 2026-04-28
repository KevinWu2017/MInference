# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
import csv
import gc
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


# DEFAULT_CONTEXT_WINDOWS = [l * 1000 for l in [10, 50, 100, 200, 300, 500, 1000]]
DEFAULT_CONTEXT_WINDOWS = [l * 1000 for l in [10]]
DEFAULT_METHODS = ["dense", "a_shape", "minference", "inf_llm", "flexprefill"]
DEFAULT_MINFERENCE_BACKENDS = ["triton", "cutlass", "v5"]
DEFAULT_MINFERENCE_BLOCK_SIZES = [32, 64]


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    attn_type: str
    backend: Optional[str] = None
    block_size: Optional[int] = None
    attn_kwargs: Optional[Dict] = None


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_context_windows(value: str) -> List[int]:
    windows = []
    for item in parse_csv_list(value):
        lower = item.lower()
        if lower.endswith("k"):
            windows.append(int(float(lower[:-1]) * 1000))
        elif lower.endswith("m"):
            windows.append(int(float(lower[:-1]) * 1000 * 1000))
        else:
            windows.append(int(item))
    return windows


def parse_int_list(value: str) -> List[int]:
    return [int(item) for item in parse_csv_list(value)]


def default_prompt_path() -> Path:
    return Path(__file__).resolve().parents[2] / "prompt_hardest.txt"


def build_cases(args) -> List[BenchmarkCase]:
    methods = parse_csv_list(args.methods)
    cases = []
    for method in methods:
        if method == "minference":
            for backend in parse_csv_list(args.minference_backends):
                for block_size in parse_int_list(args.minference_block_sizes):
                    cases.append(
                        BenchmarkCase(
                            name=f"minference/{backend}/bs{block_size}",
                            attn_type="minference",
                            backend=backend,
                            block_size=block_size,
                            attn_kwargs={
                                "minference_prefill_backend": backend,
                                "minference_prefill_block_size": block_size,
                            },
                        )
                    )
        elif method == "inf_llm":
            cases.append(
                BenchmarkCase(
                    name="inf_llm",
                    attn_type="inf_llm",
                    attn_kwargs={"dense_decoding": False},
                )
            )
        else:
            cases.append(BenchmarkCase(name=method, attn_type=method, attn_kwargs={}))
    return cases


def load_prompt_tokens(tokenizer, prompt_path: Path) -> List[int]:
    prompt = prompt_path.read_text().strip()
    if not prompt:
        raise ValueError(f"Prompt file is empty: {prompt_path}")
    return tokenizer(prompt)["input_ids"]


def make_inputs(tokenizer, prompt_tokens: List[int], context_window: int):
    repeat = context_window // len(prompt_tokens) + 1
    input_ids = (prompt_tokens * repeat)[:context_window]
    prompt = tokenizer.decode(input_ids)
    data = tokenizer(prompt, return_tensors="pt")
    return data["input_ids"].cuda(), data["attention_mask"].cuda()


def load_model(model_name: str, args, case: BenchmarkCase, kv_cache_cpu: bool):
    from minference import MInference

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
        _attn_implementation="flash_attention_2",
    )
    if case.attn_type != "hf":
        patch = MInference(
            case.attn_type,
            model_name,
            config_path=args.config_path,
            kv_cache_cpu=kv_cache_cpu,
            attn_kwargs=case.attn_kwargs or {},
        )
        model = patch.patch_model(model)
    model.eval()
    return model


def run_model_once(model, input_ids, attention_mask, attn_type: str):
    with torch.no_grad():
        if attn_type == "inf_llm":
            model.generate(
                input_ids,
                attention_mask=attention_mask,
                generation_config=GenerationConfig(max_new_tokens=1),
            )
        else:
            model(
                input_ids,
                attention_mask,
                use_cache=False,
                logits_to_keep=1,
            )


def export_torch_profile(
    model,
    input_ids,
    attention_mask,
    case: BenchmarkCase,
    context_window: int,
    profile_dir: Path,
):
    from torch.profiler import ProfilerActivity, profile

    profile_dir.mkdir(parents=True, exist_ok=True)
    trace_path = profile_dir / f"{case.name.replace('/', '_')}_{context_window}.json"
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        run_model_once(model, input_ids, attention_mask, case.attn_type)
    prof.export_chrome_trace(str(trace_path))
    return str(trace_path)


def run_target_length(
    tokenizer,
    prompt_tokens: List[int],
    model,
    case: BenchmarkCase,
    context_window: int,
    args,
) -> Dict:
    input_ids, attention_mask = make_inputs(tokenizer, prompt_tokens, context_window)

    for _ in range(args.warmup):
        torch.cuda.synchronize()
        run_model_once(model, input_ids, attention_mask, case.attn_type)
        torch.cuda.synchronize()

    trace_path = None
    if args.torch_profile:
        torch.cuda.synchronize()
        trace_path = export_torch_profile(
            model,
            input_ids,
            attention_mask,
            case,
            context_window,
            Path(args.profile_dir),
        )
        torch.cuda.synchronize()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        run_model_once(model, input_ids, attention_mask, case.attn_type)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    del input_ids, attention_mask

    mean_sec = statistics.mean(times)
    std_sec = statistics.stdev(times) if len(times) > 1 else 0.0
    return {
        "status": "ok",
        "mean_sec": mean_sec,
        "std_sec": std_sec,
        "min_sec": min(times),
        "max_sec": max(times),
        "times_sec": times,
        "peak_allocated_gb": peak_allocated / (1024**3),
        "peak_reserved_gb": peak_reserved / (1024**3),
        "torch_profile_trace": trace_path,
    }


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def should_use_kv_cache_cpu(case: BenchmarkCase, context_window: int, args) -> bool:
    return args.kv_cache_cpu or (
        args.auto_kv_cache_cpu_threshold > 0
        and context_window >= args.auto_kv_cache_cpu_threshold
        and case.attn_type not in {"inf_llm", "hf"}
    )


def failure_result(exc: RuntimeError) -> Dict:
    return {
        "status": "runtime_error",
        "error": str(exc).replace("\n", " ")[:1000],
        "mean_sec": None,
        "std_sec": None,
        "min_sec": None,
        "max_sec": None,
        "times_sec": [],
        "peak_allocated_gb": None,
        "peak_reserved_gb": None,
        "torch_profile_trace": None,
    }


def make_row(
    case: BenchmarkCase,
    context_window: int,
    args,
    kv_cache_cpu: bool,
    result: Dict,
) -> Dict:
    return {
        "case": asdict(case),
        "case_name": case.name,
        "attn_type": case.attn_type,
        "backend": case.backend,
        "block_size": case.block_size,
        "context_window": context_window,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "kv_cache_cpu": kv_cache_cpu,
        **result,
    }


def run_case_context(tokenizer, prompt_tokens, case: BenchmarkCase, context_window: int, args):
    kv_cache_cpu = should_use_kv_cache_cpu(case, context_window, args)
    model = None
    try:
        model = load_model(args.model_name, args, case, kv_cache_cpu)
        result = run_target_length(
            tokenizer, prompt_tokens, model, case, context_window, args
        )
    except RuntimeError as exc:
        result = failure_result(exc)
        if args.stop_on_error:
            raise
    finally:
        if model is not None:
            del model
        cleanup_cuda()
    return make_row(case, context_window, args, kv_cache_cpu, result)


def write_outputs(rows: List[Dict], output_prefix: Path):
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_suffix(".json")
    tsv_path = output_prefix.with_name(output_prefix.name + "_summary.tsv")

    fieldnames = [
        "case_name",
        "attn_type",
        "backend",
        "block_size",
        "context_window",
        "repeats",
        "warmup",
        "kv_cache_cpu",
        "status",
        "mean_sec",
        "std_sec",
        "min_sec",
        "max_sec",
        "peak_allocated_gb",
        "peak_reserved_gb",
        "torch_profile_trace",
        "error",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w") as f:
        json.dump(rows, f, indent=2)

    case_names = []
    context_windows = []
    for row in rows:
        if row["case_name"] not in case_names:
            case_names.append(row["case_name"])
        if row["context_window"] not in context_windows:
            context_windows.append(row["context_window"])

    lookup = {
        (row["context_window"], row["case_name"]): row
        for row in rows
    }
    summary = [["ctx"] + case_names]
    for context_window in context_windows:
        line = [f"{context_window // 1000}K"]
        for case_name in case_names:
            row = lookup.get((context_window, case_name))
            if row is None:
                line.append("")
            elif row["status"] == "ok":
                line.append(f"{row['mean_sec']:.5f}")
            else:
                line.append("ERR")
        summary.append(line)

    with tsv_path.open("w") as f:
        f.write("\n".join("\t".join(line) for line in summary))

    print("\n".join("\t".join(line) for line in summary))
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote JSON: {json_path}")
    print(f"Wrote summary: {tsv_path}")


def validate_args(args):
    valid_methods = {"hf", "dense", "a_shape", "minference", "inf_llm", "flexprefill"}
    methods = set(parse_csv_list(args.methods))
    invalid_methods = methods - valid_methods
    if invalid_methods:
        raise ValueError(f"Unsupported methods: {sorted(invalid_methods)}")

    valid_backends = {"auto", "triton", "cutlass", "v5"}
    backends = set(parse_csv_list(args.minference_backends))
    invalid_backends = backends - valid_backends
    if invalid_backends:
        raise ValueError(f"Unsupported minference backends: {sorted(invalid_backends)}")

    block_sizes = set(parse_int_list(args.minference_block_sizes))
    invalid_block_sizes = block_sizes - {32, 64}
    if invalid_block_sizes:
        raise ValueError(f"Unsupported minference block sizes: {sorted(invalid_block_sizes)}")
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")


def main():
    parser = argparse.ArgumentParser(
        description="Detailed E2E profile sweep for MInference benchmark_e2e.py workloads."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gradientai/Llama-3-8B-Instruct-Gradient-1048k",
    )
    parser.add_argument(
        "--context_windows",
        type=str,
        default=",".join(str(v) for v in DEFAULT_CONTEXT_WINDOWS),
        help="Comma-separated context windows. Supports suffixes like 10k and 1m.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated methods: hf,dense,a_shape,minference,inf_llm,flexprefill.",
    )
    parser.add_argument(
        "--minference_backends",
        type=str,
        default=",".join(DEFAULT_MINFERENCE_BACKENDS),
        help="Comma-separated minference prefill backends.",
    )
    parser.add_argument(
        "--minference_block_sizes",
        type=str,
        default=",".join(str(v) for v in DEFAULT_MINFERENCE_BLOCK_SIZES),
        help="Comma-separated minference prefill block sizes.",
    )
    parser.add_argument("--prompt_path", type=Path, default=default_prompt_path())
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--kv_cache_cpu", action="store_true")
    parser.add_argument(
        "--auto_kv_cache_cpu_threshold",
        type=int,
        default=700_000,
        help="Enable kv_cache_cpu at or above this context window. Use 0 to disable.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--torch_profile",
        action="store_true",
        help="Export a Chrome trace for the first timed iteration of each run.",
    )
    parser.add_argument("--profile_dir", type=str, default="profile_e2e_traces")
    parser.add_argument("--output_dir", type=Path, default=Path("profile_e2e_results"))
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--reload_per_context",
        action="store_true",
        help="Reload and patch the model for every context window.",
    )
    parser.add_argument("--stop_on_error", action="store_true")
    args = parser.parse_args()
    validate_args(args)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=args.trust_remote_code
    )
    prompt_tokens = load_prompt_tokens(tokenizer, args.prompt_path)
    context_windows = parse_context_windows(args.context_windows)
    cases = build_cases(args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.model_name.replace('/', '_')}_{timestamp}"
    output_prefix = args.output_dir / run_name

    rows = []
    for case in cases:
        model = None
        loaded_kv_cache_cpu = None
        for context_window in context_windows:
            print(f"Running {case.name} ctx={context_window}")
            if args.reload_per_context:
                row = run_case_context(
                    tokenizer, prompt_tokens, case, context_window, args
                )
            else:
                kv_cache_cpu = should_use_kv_cache_cpu(case, context_window, args)
                if model is None or loaded_kv_cache_cpu != kv_cache_cpu:
                    if model is not None:
                        del model
                        cleanup_cuda()
                    model = load_model(args.model_name, args, case, kv_cache_cpu)
                    loaded_kv_cache_cpu = kv_cache_cpu
                try:
                    result = run_target_length(
                        tokenizer, prompt_tokens, model, case, context_window, args
                    )
                except RuntimeError as exc:
                    result = failure_result(exc)
                    if args.stop_on_error:
                        raise
                    del model
                    model = None
                    loaded_kv_cache_cpu = None
                    cleanup_cuda()
                row = make_row(case, context_window, args, kv_cache_cpu, result)
            rows.append(row)
            if row["status"] == "ok":
                print(
                    f"{case.name} {context_window} mean={row['mean_sec']:.5f}s "
                    f"std={row['std_sec']:.5f}s peak={row['peak_allocated_gb']:.2f}GB"
                )
            else:
                print(f"{case.name} {context_window} failed: {row.get('error', '')}")
            write_outputs(rows, output_prefix)
        if model is not None:
            del model
            cleanup_cuda()

    write_outputs(rows, output_prefix)


if __name__ == "__main__":
    main()
