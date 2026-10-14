import hashlib,json,re
from pathlib import Path
root=Path('/diancpfs/user/wanghanzhen/LiteSpec')
run=root/'tests/metrics_feature_20261014'
bench=root/'tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/logging_metrics_warmup200_steps300'
s=json.loads((bench/'summary.json').read_text());assert s['gpu_isolation_audit']['passed']
assert all(c['final_masters_exactly_equal'] and c['max_loss_abs']==0 and c['max_grad_norm_abs']==0 for c in s['comparisons'].values())
assert len(s['common_measurement_steps'])==98
assert '111 passed' in (run/'functional_checks.log').read_text()
assert '55 passed' in (run/'cuda_loss_checks.log').read_text()
assert '6 passed' in (run/'scheduling_checks.log').read_text()
public=(run/'public_config_checks.log').read_text();assert 'passed' in public
legacy=json.loads((run/'legacy_confidence_parity.json').read_text())
legacy['method']='Legacy fused diagnostic reduction versus constant training reduction; FP32 forward/confidence-gradient tolerance 1e-7, BF16 leaf gradients exact in this probe.'
assert legacy['cases'][1]['logit_gradient_max_abs']==legacy['cases'][1]['confidence_gradient_max_abs']==0
(run/'legacy_confidence_parity.json').write_text(json.dumps(legacy,indent=2))
metric_audit={}
for name in s['order']:
 metric_audit[name]={}
 for rank in [0,1]:
  rows=[json.loads(x) for x in (bench/name/'telemetry'/f'steps_rank_{rank}.jsonl').read_text().splitlines()]
  counts={key:sum(r['detailed_metric_calls'].get(key,0) for r in rows) for key in ['block_accepted_length_terms','expected_accept_len_terms']}
  expected=300 if name.startswith('all') else 6
  assert all(v==expected for v in counts.values())
  metric_audit[name][str(rank)]=dict(per_metric_calls=counts,metric_steps=[r['step'] for r in rows if r['detailed_metric_calls']])
s['metric_execution_audit']=metric_audit
stages=['load_data','prepare','forward','loss','backward','update'];pooled={}
for prefix in ['all','logging']:
 trials=[v for k,v in s['measurements'].items() if k.startswith(prefix)]
 n=sum(t['samples'] for t in trials)
 ms={stage:sum(t['stages'][stage]['mean_ms']*t['samples'] for t in trials)/n for stage in stages};total=sum(ms.values())
 pooled[prefix]={stage:dict(mean_ms=ms[stage],share_percent=ms[stage]/total*100) for stage in stages}
s['pooled_stages']=pooled
(bench/'summary.json').write_text(json.dumps(s,indent=2))
files=['litespec/core/dlite.py','litespec/core/dflash.py','litespec/core/dflash2.py','litespec/core/dspark.py','litespec/core/_dflash_common.py','litespec/core/losses.py','litespec/training/recipes/dlite.py','litespec/training/runner.py','litespec/training/config.py','docs/TRAINING.md','tests/test_metrics_logging_only.py','tests/test_training_runner.py']
(run/'final_source_hashes.json').write_text(json.dumps({f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files},indent=2))
section='''### 只在日志窗口计算指标（2026-10-14）

已实现，现有 log_interval 自动控制，无需新增配置。覆盖 DLite SFT/teacher/Stage-1/transition/Stage-2 的普通、hidden-cache、packet 路径，以及 DFlash、DFlash2、DSpark。非日志窗口返回训练 loss_terms，不计算准确率、接受长度、τ、teacher agreement、top-1 分布统计、confidence error 等诊断；CE/L1/KL/BCE、selector 目标、DSpark confidence 的 detached 接受概率目标及 loss denominator 仍按训练需求保留。

共享 runner 按 optimizer step 决定，日志窗口的每个梯度累积 microbatch 都收集指标。regular interval、scheduled final 和 partial accumulation tail 强制日志都覆盖。指标仍是该 optimizer 窗口的加权汇总；不改为单独最后一个 microbatch，也不改变日志频率。独立模型调用默认 collect_detailed_metrics=True，兼容原调用；训练 recipe 显式传入 runner 的决定。

DFlash 基础 gold probability/accuracy 和 DFlash2 selector 诊断现在也受开关控制；必须训练的 top-k/selector CE 保留，包括零 alpha warmup 的参数图。DSpark 的 confidence target 每步保留，只跳过其诊断消费者。checkpoint replay 跳过诊断，L1 top-1 maxima 移到 no_grad；编译 L1 的训练部分始终使用同一 reduction，避免日志开关切换训练内核或保存张量数量。没有引入 DSpark teacher cache。

测试：functional_checks.log 的相关检查 111 passed；cuda_loss_checks.log 中 logging on/off 的完整词表 confidence target、FP32/BF16 梯度、checkpoint 及 loss 回归 55 passed；补充 regular/tail/scheduled-final 日志窗口检查 6 passed；public_config_checks.log 通过。早期 checkpoint 保存张量数量不一致的失败记录保留，已修复并复测。新回归文件为 tests/test_metrics_logging_only.py，原 runner 实训测试扩展为日志开关两种情况。

正式训练对照：logging_metrics_warmup200_steps300/，相同代码/auto loss backend、相同编译缓存，按 all→logging→logging→all 四轮各 300 步；每轮前 200 步全部预热，使用共同的 98 个正常步骤。固定初始 FP32 master、input/anchors、batch、accumulation、loss、Muon/LR schedule。两 rank 四轮 weighted loss、grad norm 全部 300 步完全一致，最终所有 FP32 master hash 完全一致。log_interval=50 时，每种诊断每 rank 从 300 次降到 6 次（50/100/150/200/250/300），非日志步骤计数为 0。

'''
section+=f"合并窗口输入吞吐：每步指标 {s['pooled_input_tokens_per_s']['all']:.2f} → 仅日志指标 {s['pooled_input_tokens_per_s']['logging']:.2f} tokens/s，本次观察 +{s['pooled_speedup_percent']:.2f}%。GPU 稳定窗口 PID/UUID 连续采样审计通过。\n\n"
for name,m in s['measurements'].items():section+=f"- {name}：step median {m['median_ms']:.3f} ms，mean {m['mean_ms']:.3f} ms，input {m['input_tokens_per_s']:.2f} tokens/s，allocated {m['peak_allocated_gib']:.3f} GiB、reserved {m['peak_reserved_gib']:.3f} GiB。\n"
section+='\n合并两轮关键 rank 时间线（含等待、通信，非纯 kernel 时长）：\n\n'
labels=dict(load_data='加载数据',prepare='前向准备',forward='forward',loss='loss及指标',backward='backward',update='参数更新')
for stage in stages:
 a,b=pooled['all'][stage],pooled['logging'][stage]
 section+=f"- {labels[stage]}：每步指标 {a['mean_ms']:.3f} ms → 仅日志指标 {b['mean_ms']:.3f} ms；新占比 {b['share_percent']:.2f}%。\n"
section+='''
吞吐含所有正常窗口的数据等待，不删除无新编译图的慢步骤。真实训练速度结论只覆盖这条双卡 offline DLite SFT；其他算法验证了功能、目标和梯度，不据此给出真实训练加速比。

额外复现检查表明，即使 torch 原生 loss，同一新代码在旧/新编译缓存下前三步也可出现不同 grad norm：torch_reference_old_cache3.json 与 torch_reference_shared_cache3.json。换回旧缓存复现旧运行 norm，新缓存的 torch reference 与本次 auto 的 norm 相同。因此本功能的数值验收使用相同代码、相同编译缓存的 on/off 对照；历史跨缓存轨迹不是开关的 baseline。

legacy_confidence_parity.json 记录旧 fused diagnostic reduction 与新固定训练 reduction 的额外比较：FP32 confidence target/leaf gradient 有最大 5.96e-8 舍入差，logit gradient 与 objective 无差；BF16 leaf gradients 在 probe 中完全一致。该记录不是声称旧/新所有 FP32 值逐位一致。当前 logging on/off 的 CUDA target/gradient 对照为严格逐元素一致。

代码和复现记录在 tests/metrics_feature_20261014/；真实训练每步数据、指标调用计数、phase event 计时、final master hash 和 GPU 审计在 tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/logging_metrics_warmup200_steps300/。没有 commit/push。
'''
p=root/'tests/training_optimization_handoff.md';text=p.read_text()
marker='### 只在日志窗口计算指标（2026-10-14）';assert marker not in text
text=text.replace('本轮也没有把全框架非日志 step 跳过所有逐位置/τ 指标作为已完成项。','此处为迁移时记录；后续已实现只在日志窗口计算指标，见本文末尾。')
text=text.replace('本清单覆盖交接已有优化、本轮验证及本轮数值修复。','本清单记录只日志指标功能加入前的优化、验证和数值修复；后续新增功能见本文末尾。')
text=text.replace('- AdamW 没有启用 fused=True；没有全框架跳过非日志逐位置/τ 指标，也没有全面的 grad scale/clip/copy kernel 融合。','- AdamW 没有启用 fused=True；grad scale/clip/copy 尚未全面融合。本条记录之后已实现非日志窗口跳过诊断指标，详见后续小节。')
text=text.replace('**最新吞吐验收要求（2026-10-14）**：','**最新功能（2026-10-14）**：只在日志窗口计算指标已实现并通过四轮 300 步同缓存对照，loss/grad norm/最终 FP32 参数完全一致；实测吞吐 +1.87%，详见本文末尾。\n\n**最新吞吐验收要求（2026-10-14）**：',1)
p.write_text(text+'\n\n'+section)
(run/'post_checks_complete.json').write_text(json.dumps(dict(complete=True,checks=['functional111','cuda_loss55','scheduling6','public_config','legacy_FP32_rounding_bound_and_BF16_leaf_gradients','same_cache_training_ABBA'],benchmark_summary=str(bench/'summary.json')),indent=2))
print(json.dumps(dict(handoff=str(p),speedup_percent=s['pooled_speedup_percent'],pooled_stages=pooled),indent=2))
