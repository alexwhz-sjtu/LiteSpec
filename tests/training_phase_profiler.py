"""Exclusive stream-timeline/host attribution without per-stage synchronization."""
import functools
import json
import time
import torch

STAGES = ('load_data', 'prepare', 'forward', 'loss', 'backward', 'update')

class PhaseProfiler:
    def __init__(self, session, runner, output, rank, warmup, nsys_window):
        self.warmup = warmup
        self.window = tuple(map(int, nsys_window.split(':'))) if nsys_window else None
        self.records = (output / f'phases_rank_{rank}.jsonl').open('w', buffering=1)
        self.enabled = False
        self.in_backward = False
        self.phase = None
        self.install(session, runner)

    def boundary(self, name):
        if not self.enabled or name == self.phase:
            return
        now = time.perf_counter()
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        if self.phase is not None:
            torch.cuda.nvtx.range_pop()
            self.intervals.append((self.phase, self.event, event, (now-self.host_start)*1000))
        self.phase, self.event, self.host_start = name, event, now
        if name is not None:
            torch.cuda.nvtx.range_push('train/' + name)

    def begin(self, step):
        self.step = step
        self.enabled = step > self.warmup
        self.intervals = []
        self.phase = None
        self.profiled = bool(self.window and self.window[0] <= step <= self.window[1])
        if self.window and step == self.window[0]:
            torch.cuda.profiler.start()
        if self.profiled:
            torch.cuda.nvtx.range_push(f'train/step_{step}')
        self.boundary('load_data')

    def end_load(self):
        self.boundary('instrumentation')

    def begin_training(self):
        self.boundary('prepare')

    def end_training(self):
        self.boundary(None)

    def collect(self, graphs, logging, wall_ms):
        if self.enabled:
            gpu = {name: 0.0 for name in (*STAGES, 'instrumentation')}
            host = dict(gpu)
            for name, start, end, cpu_ms in self.intervals:
                gpu[name] += start.elapsed_time(end)
                host[name] += cpu_ms
            self.records.write(json.dumps(dict(step=self.step, stream_timeline_ms=gpu,
                host_ms=host, rank_step_ms=wall_ms, profiled=self.profiled,
                logging_step=logging, new_graphs=graphs))+'\n')
        if self.profiled:
            torch.cuda.nvtx.range_pop()
        if self.window and self.step == self.window[1]:
            torch.distributed.barrier()
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
        self.enabled = False

    def install(self, session, runner):
        online = session.draft_online
        def wrap(obj, attr, stage, exit_stage=None, restore=False):
            original = getattr(obj, attr)
            @functools.wraps(original)
            def measured(*args, **kwargs):
                if self.in_backward or not self.enabled:
                    return original(*args, **kwargs)
                previous = self.phase
                self.boundary(stage)
                try:
                    return original(*args, **kwargs)
                finally:
                    if restore:
                        self.boundary(previous)
                    elif exit_stage is not None:
                        self.boundary(exit_stage)
            setattr(obj, attr, measured)
        wrap(online, 'prepare_batch', 'prepare')
        wrap(online, 'forward_backbone', 'forward', exit_stage='loss')
        wrap(online, 'compute_supervised_loss', 'loss')
        head = online.draft_model.sequential_head
        wrap(head, 'forward_teacher_forcing', 'forward', restore=True)
        wrap(head, 'project_logits', 'forward', restore=True)
        original_backward = runner.OptimizerWindow.backward
        @functools.wraps(original_backward)
        def backward(*args, **kwargs):
            self.boundary('backward')
            self.in_backward = True
            try:
                return original_backward(*args, **kwargs)
            finally:
                self.in_backward = False
        runner.OptimizerWindow.backward = backward
        wrap(runner.OptimizerWindow, 'step', 'update')
