from pathlib import Path
root=Path('/diancpfs/user/wanghanzhen/LiteSpec')
p=root/'docs/TRAINING.md';s=p.read_text()
needle='Logs aggregate the entire optimizer window.'
assert s.count(needle)==1
s=s.replace(needle,'''Logs aggregate the entire optimizer window.
Accuracy, acceptance length, distribution diagnostics, and loss-component
telemetry are computed only in windows that will be logged, according to
`log_interval`. Scheduled final logs and forced partial-window tail logs collect
metrics from every microbatch in that window. Ordinary non-log windows retain
only the training loss numerator/denominator; CE/L1/KL, selector objectives,
and DSpark confidence targets are still computed for training. Checkpoint
replay skips diagnostics. Direct model calls keep `collect_detailed_metrics=True`
by default; the shared runner passes the logging decision to every recipe.''')
p.write_text(s)
p=root/'litespec/training/config.py';s=p.read_text()
old='''            "Log every N optimizer steps. Metric numerators and denominators "
            "are accumulated over the complete gradient-accumulation window."'''
new='''            "Log and compute diagnostics every N optimizer steps. Metrics "
            "cover the complete logged gradient-accumulation window, including "
            "scheduled final and forced partial-window tail logs."'''
assert s.count(old)==1;p.write_text(s.replace(old,new))
