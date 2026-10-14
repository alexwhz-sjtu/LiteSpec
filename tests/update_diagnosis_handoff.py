import json
from pathlib import Path
root=Path('/diancpfs/user/wanghanzhen/LiteSpec')
base=root/'tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu'
p=json.loads((base/'phase_diagnosis_20261014/phase_summary.json').read_text())
n=json.loads((base/'numeric_diagnosis_20261014/torch/numeric/component_probe.json').read_text())
boundary=json.loads((base/'numeric_diagnosis_20261014/ce_boundary_probe.json').read_text())
snap=json.loads((base/'numeric_diagnosis_20261014/snapshot_comparison.json').read_text())
f=json.loads((base/'fixed_ce_warmup200_steps300/summary.json').read_text())
assert all(c['final_masters_exactly_equal'] for c in f['comparisons'].values())
assert f['gpu_isolation_audit']['passed']
assert len(f['common_measurement_steps'])==98
checks=json.loads((base/'numeric_diagnosis_20261014/final_related_checks.json').read_text());assert checks['exit_code']==0
stages=['load_data','prepare','forward','loss','backward','update']
labels={'load_data':'加载数据','prepare':'forward 前准备','forward':'forward','loss':'loss 计算及指标','backward':'backward','update':'参数更新'}
def aggregate(prefix):
 trials=[v for k,v in f['measurements'].items() if k.startswith(prefix)]
 count=sum(t['samples'] for t in trials)
 ms={s:sum(t['stages'][s]['mean_ms']*t['samples'] for t in trials)/count for s in stages};total=sum(ms.values())
 return {s:dict(mean_ms=ms[s],share_percent=100*ms[s]/total) for s in stages}
fixed=aggregate('auto');native=aggregate('torch')
f['pooled_stages']=dict(auto=fixed,torch=native)
(base/'fixed_ce_warmup200_steps300/summary.json').write_text(json.dumps(f,indent=2))
lines=[
'### 稳定训练分阶段计时、CUDA Graph 与数值差异定位（2026-10-14）',
'',
'本轮已完成六环节计时、稳定期 Nsight kernel 审计、真实 loss 输入及完整模型反事实实验，并修复 CE 编译导致的 BF16 梯度舍入差异。所有脚本、原始数据和报告均保存在 `tests/`；没有 commit/push。',
'',
'**配置及计时方法。** 单机 2×L20Z，Torch 2.9.1+cu130，Qwen3-8B full hidden cache；训练选取 hidden 0/8/16/24/31，final-norm 35 用于冻结 target head 投影。保持 batch=1、accumulation=1、512 anchors、block=8、5 层 draft、RNN rank=320、CE=0.1/L1=1、Muon/AdamW、seed=42、原采样顺序和 13,704 步 LR schedule。每轮先运行 200 步，再统计 201–300 的共同正常步骤，排除日志 250/300；没有删除其他慢步骤。',
'',
'六段采用同一 CUDA stream 上的共享边界 event，不在各段之间 synchronize；它们是训练关键时间线的归属（包括 CPU 导致的空闲、依赖和通信等待），不是纯 kernel 运算时间。CPU 主线程跨度另存。每步选整步较慢的 rank，再汇总该 rank 六段，因此同一份占比能相加为 100%。主线程 `next(data)` 暴露等待计入加载数据，后台 worker 读取/解包与此前计算重叠，不能把它的耗时再加到整步上。',
'',
'准备段包括 H2D、anchor/hidden gather、冻结 target LM-head 从缓存 final-norm 投影 logits、draft prepare_batch 和进入 backbone 前的 FSDP all-gather。forward 包括 backbone、RNN 和 draft logits 投影；后两者虽写在 compute_supervised_loss 内，已单独归入 forward。loss 包括 CE/L1、mask/权重/归一化和当前每步指标。backward 包括反向通信以及发生于反向中的重算；update 包括梯度归一化、norm/clip、Muon/AdamW、scheduler 和 copy-back。',
'',
'**修复前分段基线。** `phase_diagnosis_20261014/phase_summary.json`：auto 95 个样本（额外排除 Nsight 的 240–242），torch 98 个样本。auto 平均 wall=236.03 ms、六段合计=235.66 ms，额外 fingerprint=0.278 ms、边界及最终同步余量=0.091 ms。阶段钩子的初始化、全部 300 步 input/anchor、loss 和 grad norm 与之前各自 backend 的基线完全一致。',
]
for s in stages:
 a=p['phase_summary']['auto']['stages'][s];t=p['phase_summary']['torch']['stages'][s]
 lines.append(f"- {labels[s]}：auto {a['mean_stream_ms']:.3f} ms / {a['share_percent']:.2f}%；torch {t['mean_stream_ms']:.3f} ms / {t['share_percent']:.2f}%。")
lines += [
'',
'**CUDA kernel 与 Graph 实验。** `phase_diagnosis_20261014/stable_240_242.nsys-rep` 仅捕获预热后的 240–242 步；两卡共 13,179 个 kernel，约 2,196.5 个/卡/步。5,112 个（38.79%）低于 10µs，6,631 个（50.31%）低于 20µs。低于 10µs 的 kernel 只占跨卡/stream 累计 kernel 时长 1.11%，此累计量不能当整步 wall time。CUDA Graph API 次数和非零 graphNodeId 都为 0。',
'',
'按 launch 所属 CPU NVTX range + process/correlationId 归属（包括 autograd 工作线程），update 4,472 个 kernel、backward 4,908、forward 2,665、loss 762、prepare 348；仅 24 个在六段之外。update 中有大量 dtype cast/copy、逐参数 scale、norm、momentum/weight decay 等；较重的 Muon Newton-Schulz GEMM 和 NCCL SendRecv 也在 update。因此更新约 46% 并不等于这 46% 全部可以通过减少小 kernel 消除。',
'',
'当前完整 step 有动态 active-position 形状、数据加载、CPU 分支，以及 `normalize_accumulated_gradients` 的 denominator.item、BF16Optimizer 的 active-mask.tolist / grad_norm.item；不能直接捕获整步。优先候选是静态形状 Newton-Schulz 子过程、固定 buffer 的参数 cast/scale/copy，以及可保持原梯度舍入边界的 foreach/pointwise 融合。CUDA Graph 是捕获并批量重放 launch，不等同把多个数学 kernel 融成一个。',
'',
'`phase_diagnosis_20261014/muon_cudagraph_probe.json` 已实际验证纯 CUDA Newton-Schulz 可 capture/replay：每种 shape 200 次调用预热、eager/graph/graph/eager 各 100 次；输出逐元素完全一致。640×640：约 0.271→0.096 ms/次（约 64%）；4096×4096：3.544→3.357 ms（约 5.3%）；151936×320：1.904→1.867 ms（约 1.9%）。这是子过程 microbenchmark，尚未接入正式 optimizer，不声称整步获得上述加速。大矩阵计算和通信仍占主要时间。',
'',
'**数值差异的因果定位。** 修复前相同初始化、300 步 input/anchor；同 backend 重复 loss 完全一致，但跨 backend 第 1 步 loss 相同、grad norm 已相差约 3.0e-6，第 2 步 loss 开始分叉，最大到 0.015278。',
'',
'真实第 1 步 rank 0 的 loss 输入 shape=[1127,151936]，draft logits 是 BF16 head 输出后 cast 成 FP32，teacher 为 BF16。`numeric_diagnosis_20261014/torch/numeric/component_probe.json`：单独 L1 的 FP32 梯度完全一致；compiled CE 的 FP32 梯度最大差 4.55e-12，8,659,927 元素不同，其中 1,316 个在转为 BF16 后仍不同。CE+L1 组合仍有 924 个 BF16 梯度不同。原保护只检查 loss 接口 logits.dtype，因此这条已 cast 为 FP32 的真实训练路径仍进入 compiled CE。',
'',
'进一步的 `ce_boundary_probe.json` 使用同一真实输入的 32 行定位到 log-softmax：native log-softmax + native NLL 与 CE 基准一致；compiled log-softmax + native NLL 复现 compiled CE 的梯度差异；native log-softmax + compiled gather/reduction 的梯度完全一致。compiled log-softmax 的输出最大差 1.91e-6。Inductor 生成源码展示分段 max/sum 归约和 Triton log-softmax，这条不同的浮点实现导致 FP32 梯度落在 BF16 舍入边界两侧，进而改变网络反向及后续优化轨迹。不能因为 scalar loss close 就认为 BF16 参数更新相同。',
'',
'完整网络的 first-step snapshots 见 `numeric_diagnosis_20261014/snapshot_comparison.json`：旧 auto 在两 rank 分别有 44,735,412 / 16,771,675 个 BF16 model-gradient 元素不同；FP32 master 更新后分别有 239,639,681 / 212,126,267 个元素不同，其中最终转 BF16 后仍有 36,382 / 25,053 个不同。只恢复 native CE、保持 compiled L1 的反事实实验，两 rank 全部 model gradients 和全部 FP32 master 更新均逐元素一致。',
'',
'**修复和复测。** `litespec/core/losses.py` 保留所有 dtype 的 native CE forward/backward，继续编译 native softmax 后的 L1 reduction；public weighted_cross_entropy 同样保护 FP32 来自 BF16 的梯度边界。没有修改 loss 数学公式、数据、anchors 或 optimizer。`tests/test_loss_fusion.py` 新增真实词表尺寸、BF16 head→FP32 loss 的 CE/CE+L1 精确梯度回归，两例在修复前失败、修复后通过；loss suite 33 passed；projection/recompute、Stage-1 loss、runner 相关测试 74 passed，记录见 `related_training_tests_gpu_available.log`。',
'',
'修复后 `fixed_ce_warmup200_steps300/summary.json`：torch→auto→auto→torch 四轮各 300 步，前 200 步全部预热，共同正常样本 98 个/轮。两 rank 四轮 initial master、input 和 anchors 相同；全部 300 步 grad norm 完全一致，最终全部 FP32 master 的 SHA-256 四轮完全一致。跨 backend weighted loss 最大差 '+f"{max(c['max_loss_abs'] for c in f['comparisons'].values()):.10g}"+'，来自 compiled L1 求和的浮点舍入，已不产生参数轨迹分叉。',
'',
'**当前修复后的正式六环节占比**（合并两轮 auto，共 196 个稳定样本）：',
]
for s in stages:
 a=fixed[s];t=native[s]
 lines.append(f"- {labels[s]}：auto {a['mean_ms']:.3f} ms / {a['share_percent']:.2f}%；torch {t['mean_ms']:.3f} ms / {t['share_percent']:.2f}%。")
lines += ['',f"合并稳定窗口输入吞吐：torch {f['pooled_input_tokens_per_s']['torch']:.2f}、auto {f['pooled_input_tokens_per_s']['auto']:.2f} tokens/s，观察差异 {f['pooled_speedup_percent']:+.2f}%。当前复测没有显示可靠的整体吞吐收益，不能沿用修复前 compiled CE 的性能数字。"]
for name,m in f['measurements'].items():
 lines.append(f"- {name}：median {m['median_ms']:.3f} ms，mean {m['mean_ms']:.3f} ms，input {m['input_tokens_per_s']:.2f} tokens/s，allocated 峰值 {m['peak_allocated_gib']:.3f} GiB、reserved {m['peak_reserved_gib']:.3f} GiB。")
lines += ['', '稳定窗口 GPU UUID/driver PID 连续采样审计通过，见 `fixed_ce_warmup200_steps300/gpu_isolation_audit.json`；采样无法排除短于间隔的外部活动。本轮证明当前双卡、指定 hidden 的 offline DLite SFT；没有据此扩大到其他算法或旧多节点 online 故障。', '', '脚本入口：`tests/benchmark_training_phases.py`、`training_phase_profiler.py`、`benchmark_training_numeric.py`、`training_numeric_probe.py`、`probe_ce_boundary.py`、`probe_muon_cudagraph.py`；控制器 `tests/environment-setup/run_phase_diagnosis.py`、`run_numeric_diagnosis.py`、`run_fixed_ce_abba.py`。各 config、init/step/phase/complete telemetry、Nsight SQLite 和测试日志均保留。', '']
section='\n'.join(lines)
handoff=root/'tests/training_optimization_handoff.md';source=handoff.read_text()
marker='### 稳定训练分阶段计时、CUDA Graph 与数值差异定位（2026-10-14）'
assert marker not in source
source=source.replace('四轮各 300 步 ABBA 已完成；正式记录见本文末尾“吞吐复测：每轮预热 200 步”以及 `newhost_2gpu/warmup200_steps300/abba_summary.json`。', '六环节计时及数值根因修复已完成；最新正式记录见本文末尾“稳定训练分阶段计时、CUDA Graph 与数值差异定位”及 `newhost_2gpu/fixed_ce_warmup200_steps300/summary.json`。之前的吞吐 ABBA 保留为修复前历史记录。')
source=source.replace('跨 backend 的长期训练轨迹仍有待核验。','跨 backend 的差异已定位到 compiled CE 的 log-softmax 舍入，并修复；最新 300 步复测最终 FP32 参数逐元素一致，详见末尾数值定位小节。')
source=source.replace('差异原因尚未定位。','此为修复前记录；差异在后续小节中已定位并修复。')
handoff.write_text(source+'\n'+section)
print(json.dumps(dict(handoff=str(handoff),pooled_stages=f['pooled_stages'],speedup=f['pooled_speedup_percent']),indent=2))
