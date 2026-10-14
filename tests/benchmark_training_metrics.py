"""Measure unchanged DLite runner steps; experimental telemetry stays under tests."""
import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=5)
    trace_options = parser.add_mutually_exclusive_group()
    trace_options.add_argument("--trace-step", type=int)
    trace_options.add_argument("--nsys-trace-step", type=int)
    parser.add_argument("--save-final", action="store_true")
    parser.add_argument("--verified-cache-record", type=Path, help="Reuse a completed checksum audit while requiring unchanged cache files")
    parser.add_argument("--stage-nsys-window")
    parser.add_argument("--metrics-every-step", action="store_true")
    parser.add_argument("--final-masters-hash", action="store_true")
    values = parser.parse_args()
    import torch
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    from litespec.training.runner import configure_rank_caches
    configure_rank_caches()
    import torch.distributed as dist
    from torch._dynamo.utils import counters
    from accelerate.utils import set_seed
    from scripts.config.launch_dlite import load_config
    from scripts.config.training_launcher import training_base_cli
    from litespec.training.algorithms import get_training_adapter
    from litespec.training.setup import build_training_session
    from litespec.training import runner
    from litespec.distributed import destroy_distributed

    configured_env, cli = load_config(values.config, "sft")
    os.environ.update({k: v for k, v in configured_env.items() if k not in ("MASTER_ADDR", "MASTER_PORT")})
    adapter = get_training_adapter("dlite")
    args = adapter.parse_args("sft", argv=training_base_cli(configured_env) + cli)
    if args.accumulation_steps != 1:
        raise ValueError("This telemetry harness expects the handoff's accumulation_steps=1")
    args.max_steps = values.steps
    args.final_save = values.save_final
    set_seed(args.seed)
    if values.verified_cache_record:
        import litespec.training.setup as training_setup
        verification = json.loads(values.verified_cache_record.read_text())
        if not verification.get("complete") or not verification.get("all_shard_checksums_verified"):
            raise ValueError("Cache verification record is incomplete")
        verified_root = Path(verification["cache_root"]).resolve()
        if not args.train_hidden_states_path or Path(args.train_hidden_states_path).resolve() != verified_root:
            raise ValueError("Cache verification record belongs to another training input")
        def unchanged_cache():
            for relative, expected in verification["verified_file_stats"].items():
                item = (verified_root / relative).stat()
                observed = dict(size=item.st_size, mtime_ns=item.st_mtime_ns, ctime_ns=item.st_ctime_ns, inode=item.st_ino, device=item.st_dev)
                if observed != expected:
                    raise RuntimeError(f"Verified cache file changed: {relative}")
            digest = hashlib.sha256((verified_root / "manifest.json").read_bytes()).hexdigest()
            if digest != verification["manifest_sha256"]:
                raise RuntimeError("Verified cache manifest changed")
        original_manifest_loader = training_setup.load_hidden_cache_manifest
        def verified_manifest_loader(cache_root, **keywords):
            if Path(cache_root).resolve() != verified_root:
                return original_manifest_loader(cache_root, **keywords)
            unchanged_cache()
            keywords["verify_checksums"] = False
            result = original_manifest_loader(cache_root, **keywords)
            unchanged_cache()
            return result
        training_setup.load_hidden_cache_manifest = verified_manifest_loader
    session = None
    completed = False
    try:
        session = build_training_session(args, adapter)
        if values.metrics_every_step:
            original_compute = session.adapter.compute_step
            def all_metrics(*positional, **keywords):
                keywords["collect_detailed_metrics"] = True
                return original_compute(*positional, **keywords)
            session.adapter.compute_step = all_metrics
        rank = dist.get_rank()
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        telemetry = output / "telemetry"
        telemetry.mkdir(exist_ok=True)
        initial = hashlib.sha256()
        for param in session.optimizer.fp32_params:
            initial.update(str(tuple(param.shape)).encode())
            initial.update(param.detach().float().cpu().numpy().tobytes())
        gpu_uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
        if not gpu_uuid.startswith("GPU-"):
            gpu_uuid = "GPU-" + gpu_uuid
        query = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
        driver_pids = [int(line.split(",")[0]) for line in query.stdout.splitlines() if line.split(",")[-1].strip() == gpu_uuid]
        if len(driver_pids) != 1:
            raise RuntimeError(f"Expected one training process on {gpu_uuid}: {driver_pids}")
        info = {
            "rank": rank, "container_pid": os.getpid(),
            "gpu_process_identity": {"driver_pid": driver_pids[0], "gpu_uuid": gpu_uuid}, "seed": args.seed, "metrics_every_step_for_test": values.metrics_every_step,
            "initial_master_sha256": initial.hexdigest(), "backend": os.environ.get("LITESPEC_LOSS_BACKEND", "auto"),
            "cache_verification_record": str(values.verified_cache_record) if values.verified_cache_record else None,
            "args": vars(args), "steps_per_epoch": session.steps_per_epoch,
            "schedule_steps": session.total_steps,
        }
        (telemetry / f"init_rank_{rank}.json").write_text(json.dumps(info, indent=2))
        current = {}
        anchors = []
        if values.trace_step is not None or values.nsys_trace_step is not None:
            original_project = session.draft_online.draft_model.sequential_head.project_logits
            def project(*positional, **keywords):
                if current.get("profiling", False):
                    current["draft_projection_calls"] = current.get("draft_projection_calls", 0) + 1
                    with torch.profiler.record_function("dlite/draft_project_logits"), torch.cuda.nvtx.range("dlite/draft_project_logits"):
                        return original_project(*positional, **keywords)
                return original_project(*positional, **keywords)
            session.draft_online.draft_model.sequential_head.project_logits = project
        import litespec.core.dlite as dlite_core
        for name in ("block_accepted_length_terms", "expected_accept_len_terms"):
            original = getattr(dlite_core, name)
            def measured_metric(*positional, _name=name, _original=original, **keywords):
                calls = current.setdefault("detailed_metric_calls", {})
                calls[_name] = calls.get(_name, 0) + 1
                return _original(*positional, **keywords)
            setattr(dlite_core, name, measured_metric)
        original_sample = session.draft_online.sample_anchor_positions
        def sample(*positional, **keywords):
            result = original_sample(*positional, **keywords)
            anchors.append(tuple(t.detach() for t in result))
            return result
        session.draft_online.sample_anchor_positions = sample
        original_backward = runner.OptimizerWindow.backward
        def backward(window, result):
            current["numerator"] = result.loss_terms.numerator.detach()
            current["denominator"] = result.loss_terms.denominator.detach()
            return original_backward(window, result)
        runner.OptimizerWindow.backward = backward
        original_step = runner.OptimizerWindow.step
        def step(window):
            norm = original_step(window)
            current["grad_norm"] = float(norm)
            return norm
        runner.OptimizerWindow.step = step
        def log(*positional, **keywords):
            current["logging_step"] = True
            return runner.log_window(*positional, **keywords)

        from training_phase_profiler import PhaseProfiler
        phase_profiler = PhaseProfiler(session, runner, telemetry, rank, values.warmup_steps, values.stage_nsys_window)

        def fingerprint(tensor):
            return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

        def batches(session, phase):
            source = iter(runner.iter_logical_batches(session, phase))
            with (telemetry / f"steps_rank_{rank}.jsonl").open("w", buffering=1) as records:
                while True:
                    dist.barrier()
                    torch.cuda.synchronize()
                    wall_started = time.time()
                    started = time.perf_counter()
                    phase_profiler.begin(session.state.optimizer_step + 1)
                    try:
                        data = next(source)
                    except StopIteration:
                        return
                    phase_profiler.end_load()
                    current.clear()
                    anchors.clear()
                    input_hash = fingerprint(data["input_ids"])
                    input_tokens = int(data.get("attention_mask", torch.ones_like(data["input_ids"])).sum())
                    supervised_tokens = int((data["loss_mask"] > 0.5).sum())
                    upcoming = session.state.optimizer_step + 1
                    profiler = None
                    if upcoming == values.trace_step:
                        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True, profile_memory=True)
                        profiler.start()
                        current["profiling"] = True
                    nsys_profiling = upcoming == values.nsys_trace_step
                    if nsys_profiling:
                        torch.cuda.profiler.start()
                        torch.cuda.nvtx.range_push(f"dlite/step_{upcoming}")
                        current["profiling"] = True
                    graphs_before = counters["stats"]["unique_graphs"]
                    phase_profiler.begin_training()
                    yield data
                    phase_profiler.end_training()
                    new_graphs = counters["stats"]["unique_graphs"] - graphs_before
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    phase_profiler.collect(new_graphs, current.get("logging_step", False), elapsed*1000)
                    if nsys_profiling:
                        dist.barrier()
                        torch.cuda.synchronize()
                        torch.cuda.nvtx.range_pop()
                        torch.cuda.profiler.stop()
                    if profiler is not None:
                        profiler.stop()
                        profiler.export_chrome_trace(str(telemetry / f"trace_rank_{rank}_step_{upcoming}.json"))
                        (telemetry / f"profile_rank_{rank}.txt").write_text(profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=80))
                    numerator = float(current["numerator"])
                    denominator = float(current["denominator"])
                    grad_norm = current.get("grad_norm")
                    if not all(math.isfinite(value) for value in [numerator, denominator, grad_norm]):
                        raise RuntimeError(f"Non-finite telemetry at step {upcoming}")
                    sums = torch.tensor([numerator, denominator, input_tokens, supervised_tokens], device="cuda", dtype=torch.float64)
                    dist.all_reduce(sums)
                    slowest = torch.tensor([elapsed, new_graphs], device="cuda", dtype=torch.float64)
                    dist.all_reduce(slowest, op=dist.ReduceOp.MAX)
                    record = {
                        "step": session.state.optimizer_step, "rank": rank,
                        "wall_time_started": wall_started, "wall_time_ended": time.time(),
                        "rank_step_ms": elapsed*1000, "slowest_rank_step_ms": slowest[0].item()*1000,
                        "global_weighted_loss": (sums[0] / sums[1]).item(),
                        "global_denominator": sums[1].item(),
                        "global_input_tokens": int(sums[2].item()), "global_supervised_tokens": int(sums[3].item()),
                        "grad_norm": grad_norm, "input_sha256": input_hash,
                        "anchors_sha256": [fingerprint(t) for pair in anchors for t in pair],
                        "logging_step": current.get("logging_step", False),
                        "detailed_metric_calls": current.get("detailed_metric_calls", {}),
                        "draft_projection_calls_if_profiled": current.get("draft_projection_calls"),
                        "new_graphs_on_any_rank": int(slowest[1].item()),
                        "profiled": profiler is not None or nsys_profiling, "warmup": upcoming <= values.warmup_steps,
                        "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                        "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
                    }
                    records.write(json.dumps(record)+"\n")
                    if rank == 0:
                        print("STEP_TELEMETRY " + json.dumps(record), flush=True)
        runner.run_draft(session, batch_source=batches, log=log)
        dist.barrier()
        final_master_hash = None
        if values.final_masters_hash:
            digest = hashlib.sha256()
            for param in session.optimizer.fp32_params:
                digest.update(str(tuple(param.shape)).encode())
                digest.update(param.detach().float().cpu().numpy().tobytes())
            final_master_hash = digest.hexdigest()
        (telemetry / f"complete_rank_{rank}.json").write_text(json.dumps({"optimizer_steps": session.state.optimizer_step, "max_steps": values.steps, "schedule_steps": session.total_steps, "final_master_sha256": final_master_hash}))
        completed = True
    finally:
        if session is not None and session.tracker is not None:
            session.tracker.close()
        if completed:
            destroy_distributed()

if __name__ == "__main__":
    main()
