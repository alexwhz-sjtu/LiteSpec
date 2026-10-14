# LiteSpec 训练优化交接文档

记录日期：2026-10-13。

**最新功能（2026-10-14）**：只在日志窗口计算指标已实现并通过四轮 300 步同缓存对照，loss/grad norm/最终 FP32 参数完全一致；实测吞吐 +1.87%，详见本文末尾。

**最新吞吐验收要求（2026-10-14）**：用户要求每轮先运行 200 steps，吞吐仅在第 201 步及之后统计。此前 60 步、5 步预热的 2.3% 是初步观察，不作为正式性能结论。六环节计时及数值根因修复已完成；最新正式记录见本文末尾“稳定训练分阶段计时、CUDA Graph 与数值差异定位”及 `newhost_2gpu/fixed_ce_warmup200_steps300/summary.json`。之前的吞吐 ABBA 保留为修复前历史记录。


本文保留原迁移交接（第 1–8 节），并续写当前工作树在新机器上的验证（第 9 节及后续小节）。full cache、双卡 450 步稳定性、Nsight GPU profile 和每轮预热 200 步的吞吐复测已完成。旧机器受干扰的计时及新机器最初的 60 步结果不作为正式吞吐结论；跨 backend 的差异已定位到 compiled CE 的 log-softmax 舍入，并修复；最新 300 步复测最终 FP32 参数逐元素一致，详见末尾数值定位小节。

## 1. 用户已经确定的范围

- 优化范围覆盖 DLite、DFlash、DFlash2、DSpark 的相关公共训练路径。
- 不做长度分桶，不改变 batch size、采样顺序或数据组织。
- 处理 DLite transition 重复 head、DFlash 系列日志重复投影、FP32→BF16 显式临时张量，以及公共 loss 融合。
- **不要给 DSpark 增加 teacher 投影缓存。** 曾实现的 `_TeacherReplayCache` 和 `LITESPEC_TEACHER_CACHE_MIB` 已按用户要求删除，对应缓存测试也已删除。后续不要把它们作为现有功能恢复。
- 不以共享 GPU 上的计时得出加速结论；先保证数值，再在空闲机器上验证收益。

此前还讨论过 fused AdamW、grad norm、full-attention mask 和 ptxas 探测。本文只将核对过的修改列为完成项：当前 `BF16Optimizer` 的 `torch.optim.AdamW(...)` **没有显式传入 `fused=True`**；没有据此声称已启用 fused AdamW。此处为迁移时记录；后续已实现只在日志窗口计算指标，见本文末尾。

## 2. 迁移版本与文件清单

最终核对时 HEAD：**`afbb9ed`，提交标题 `update kernel and loss ops`**。整理开始时 HEAD 是 `0e65148`，部分修改和测试尚未提交；整理期间出现的新提交 `afbb9ed` 已将它们及最新重算记录纳入版本控制。较早的相关提交还包括 `8fc4ab8`、`5ee551a`。

迁移时以 `afbb9ed` 为源码基线，并另行带上本文。最终核对时仅本交接文档仍未跟踪；本文没有替用户执行 commit 或 push，也未确认该提交已经推送到远端。请确保新机器能取得该提交，不能只迁移一个相对于它的空代码 diff。

| 文件 | 作用 | 写文档时状态 |
| --- | --- | --- |
| `litespec/core/dlite.py` | transition 复用 serial logits；接入 CE/L1/KL | 已在 HEAD，最新修改见 `afbb9ed` |
| `litespec/core/dflash.py` | 日志复用 draft logits，反向重算跳过日志回调 | 已在 HEAD |
| `litespec/core/_dflash_common.py` | 日志可接收现成 logits；公共 CE | 已在 HEAD，最新修改见 `afbb9ed` |
| `litespec/core/dflash2.py` | selector CE 使用公共 loss | 已在 HEAD，最新修改见 `afbb9ed` |
| `litespec/core/dspark.py` | 公共 CE/L1/BCE；复用分布诊断结果；无 teacher 缓存 | 已在 HEAD，最新修改见 `afbb9ed` |
| `litespec/core/chunking.py` | checkpoint 重算上下文标识 | 已在 HEAD |
| `litespec/core/losses.py` | loss 参考实现、编译入口及数值保护 | 已在 HEAD，最新修改见 `afbb9ed` |
| `litespec/training/optimizer.py` | 更新和恢复时直接 `copy_` 到模型参数 | 已在 HEAD |
| `tests/test_loss_fusion.py` | loss 数值、梯度和边界测试 | 已在 `afbb9ed` |
| `tests/test_training_compute_reuse.py` | head/日志复用、参数转换测试 | 已在 `afbb9ed` |
| `tests/distributed_loss_fusion_smoke.py` | 双卡 FSDP、梯度累积、恢复检查 | 已在 `afbb9ed` |
| `tests/benchmark_loss_fusion.py` | loss 前向+反向微基准 | 已在 `afbb9ed` |
| `tests/loss_fusion_fp32_results.json` | 受干扰且属于旧实现的历史数据 | 已在 `afbb9ed` |
| `tests/inspect_objective_recompute.py` | CPU 小模型 checkpoint 重算计数 | 已在 HEAD |
| `tests/training_recompute_counts.txt` | 删除缓存后的当前重算计数实测 | 本次交接生成，已在 `afbb9ed` |
| `tests/training_optimization_handoff.md` | 本文 | 本次交接新增，未跟踪 |

这是相关文件清单，不是独立运行所需的最小源码包。测试依赖仓库中其他测试辅助文件，例如 `tests/refactor_baseline.py` 和 `tests/test_dflash_family.py`，应迁移完整源码。模型、数据及 checkpoint 的存放路径需要在新机器配置中调整；不要直接搬用旧机器 `.venv` 或把旧 Inductor 编译缓存当作可移植产物。

## 3. 最终实现的具体行为

### DLite transition

`OnlineDLiteModel.compute_supervised_loss(..., serial_logits=None)` 可以接收已经计算的 serial logits。transition 前向先得到 student serial logits，供 Stage-1 KL 和 supervised objective 共用，避免第二次 sequential head / 词表投影。两个 loss 仍通过同一张计算图向 student 累积梯度，没有把训练 logits detach。

普通 supervised 调用没有传入 logits 时，仍自行计算。测试比较了共享图和分别计算时的 loss、hidden 梯度及参数梯度，并检查 transition 只调用一次 `project_logits`。

### DFlash / DFlash2 日志

objective 的投影结果通过 `metrics_callback` 交给指标计算；`_dflash_metric_chunk_terms(..., logits=...)` 可直接使用已有 logits，不再为日志额外投影一次 draft hidden。

`chunking.py` 通过 `ContextVar` 标记 checkpoint 重算阶段，日志回调只在原始前向运行。日志在 `no_grad` 下计算，仅累加小型指标结果，不保存整块 logits 的训练图。用于日志的 teacher 投影也不会随该回调在反向重算。

这不等于“反向只重算 draft logits”：checkpoint 内 CE、selector 等实际训练 objective 运算仍可能重算。DFlash2 candidate mask、selector 权重及其训练语义继续保留。

### FP32→BF16 同步

`BF16Optimizer.step()` 中由显式转换后拷贝改为 `p.copy_(mp)`；恢复 optimizer 状态时同样用 `model_param.copy_(master)`。在已有 `no_grad` 中，由 `copy_` 完成目标 dtype 转换，避免先创建一份完整的 BF16 转换结果再拷贝。

FP32 master weights、优化器状态以及用于优化器的 FP32 梯度仍然存在。没有删除梯度裁剪或修改 Muon 的矩阵分配/通信。当前全局 grad norm 使用 `_foreach_norm`、小标量归约和一个标量 all-reduce。

### CE / L1 / KL / BCE 公共 loss

**版本说明：本段及其表格记录 afbb9ed 迁移时的策略。2026-10-14 已撤回所有 dtype 的 compiled CE，当前 CE 保持原生，仅编译适用的 L1/KL/BCE 路径；最新策略与收益见本文末尾。**

入口在 [losses.py](../litespec/core/losses.py)。以 PyTorch eager 表达式为数值参考，使用 `torch.compile(backend="inductor", fullgraph=True, dynamic=True)` 和 AOTAutograd 生成 Triton 前向/反向；没有维护一套手写梯度公式。

公共 helper 返回加权 numerator 和必要的 detached 诊断结果；原有 denominator、跨 rank 归一化、梯度累积及 objective 系数由调用方继续处理。

| 路径 | 最终策略 |
| --- | --- |
| FP32 CE | 允许编译 CE 与加权归约 |
| BF16/FP16 CE | 保留原生 CE，即使选择 `inductor` 也不强制融合此路径 |
| CE + L1 | L1 的 teacher/student softmax 均保留原生 FP32 计算；融合后续差值、绝对值、加权和归约。FP32 CE 可一并进入编译函数 |
| KL | 编译路径先计算 detached teacher log-prob/prob，再融合 student 路径及归约，避免 AOT 在其自身 backward 中重新计算 teacher normalization |
| BCE | FP32 target 时允许编译；其他 target dtype 保留参考路径 |

KL 的上述边界不保证外层 activation checkpoint 完全不重算 objective。当前按常规一阶训练 backward 验证，未验证高阶梯度；AOT 的 donated buffers 也可能限制对同一图反复 `retain_graph=True` 的用法。

**必须保留的数值保护：**

1. 当前 PyTorch 2.9.1 的编译器会改变低精度 CE 的中间舍入，即使设置 `emulate_precision_casts=True` 也不足以保持原生结果，所以低精度 CE 回退。
2. 把 L1 两侧 softmax 一并融合时，实测 teacher 与 student 完全相同仍可能产生约 `1e-7` 的非零 L1；因为 `abs` 在零点的梯度敏感，单看标量误差不足以判断安全。最终改为原生 softmax 加融合归约，完全相等和接近相等的边界梯度测试通过。不要简单扩大容差绕过此问题。
3. 数值检查通过不代表所有浮点结果逐位相同，也不代表已经完成长程收敛验证。

### DSpark 与 `objective_chunk_blocks`

DSpark 使用公共 CE/L1/BCE，并复用 L1 路径已经得到的 detached 分布距离和 top-1 概率来构造诊断指标。teacher 投影在 objective 内直接计算，**无缓存环境变量、缓存对象或缓存预算**。

- `objective_chunk_blocks > 0`：按 block 分块；梯度开启且输入需要梯度时使用 non-reentrant activation checkpoint，反向可能重算 objective，包括 DSpark 的 teacher 投影。
- `objective_chunk_blocks = 0`：整段 objective 一次计算、不使用这一层 checkpoint；正常 backward 仍执行，通常保存更多激活、占用更高峰值显存。
- draft Transformer 自身的 gradient checkpointing 是独立机制，不能由这个参数推断其已关闭。

## 4. 运行开关与环境

`LITESPEC_LOSS_BACKEND` 是进程环境变量，不是本轮新增的 YAML 字段。

| 值 | 行为 |
| --- | --- |
| `auto`，默认 | CUDA 且输入 logits 元素数至少 65536 时启用适用的编译路径；CPU/小输入走 eager |
| `torch` | 公共 loss 使用 eager 数值参考；其他 head 复用和参数拷贝优化仍生效 |
| `inductor` | CUDA 上绕过上述大小门槛，用于比较；仍保留低精度 CE、L1 softmax 等数值保护 |

65536 是当前派发门槛，并非由可靠性能实验调出的最优阈值。切换 `torch` 只隔离 loss 融合，不是回退到所有优化之前的代码。

旧机器环境实查：Python **3.11.15**、PyTorch **2.9.1+cu128**、Triton **3.5.1**、PyTorch CUDA runtime **12.8**；GPU 测试/历史测量使用 NVIDIA H800。项目 `pyproject.toml` 约束 Python 3.11 和 torch 2.9.1；迁移时结合 `uv.lock` 重建环境，并记录新 GPU/driver/软件版本。

用户此前远程训练配置为 DLite SFT、Qwen3-8B、6 nodes × 8 ranks、`tp_size=1`、`batch_size=1`、`accumulation_steps=1`、`max_length=10240`、online SGLang `fa3`、draft `flex_attention/TRITON`、Muon 开启、`disaggregate=false`。早先建议中的 MBS4 不是这份配置的实际 batch size。本轮没有完成该 48-rank 完整配置的长程验证，也没有确认此前约 400 step 的远程故障已被这些修改修复。

## 5. 已做验证及其边界

| 检查 | 会话中结果 | 说明 |
| --- | --- | --- |
| 相关广覆盖回归 | 195 passed，另有 4 subtests passed | 该次运行早于最终 L1 保护调整和缓存删除，不能当作最终工作树全量重跑记录 |
| `test_loss_fusion.py` | 31 passed | 最终 L1 保护调整后运行；覆盖 FP32/BF16、mask、teacher detach、不同形状、极端有限 logits、autocast、相等/近似相等、checkpoint |
| `test_training_compute_reuse.py` + `test_training_refactor.py` + `test_dflash_family.py` | 47 passed | 删除 DSpark 缓存后运行；当前相关回归结果 |
| 双卡 `distributed_loss_fusion_smoke.py` | PASS，观察到 max master error = 0 | FSDP `SHARD_GRAD_OP` + BF16 小线性模型，3 个更新 step，每 step 累积 2 个 micro-batch，rank 间有效权重不同，中途恢复 optimizer state |
| 双卡 `distributed_muon_smoke.py` | matrices_per_rank=1、2 均 PASS；观察到 max master error = 0 | 更新、参数 dtype 同步、momentum 和分布式 checkpoint 恢复；小模型检查 |
| CPU objective 重算计数 | DFlash、DFlash2、DSpark 均通过 loss/hidden 梯度对照 | 编写本文时在已删除缓存的当前代码上重新运行，原始输出见下方链接 |

31 项 loss 测试中的主要容差：标量/诊断 `rtol=3e-5, atol=3e-5`；FP32 梯度 `rtol=3e-5, atol=2e-6`；BF16 梯度 `rtol=0.016, atol=2e-5`，用于覆盖 dtype 的最终舍入。checkpoint 和边界测试还有独立阈值，见测试源码。双卡 smoke 的“观测误差为零”仅描述此次小模型，不能外推为真实模型始终逐位一致。

上述多数测试日志为会话记录，未保存独立完整日志文件。新机器建议将复跑输出保存在 `tests/`，不要仅沿用历史 PASS 结论。

当前 [重算原始输出](training_recompute_counts.txt) 使用 CPU eager、小词表、2 个 block，直接调用 objective；不执行完整 attention，也不覆盖 DFlash 日志回调计数：

| 算法 / chunk_size | 前向 draft 投影 | 前向 teacher 投影 | backward 重算 draft 投影 | backward 重算 teacher 投影 |
| --- | ---: | ---: | ---: | ---: |
| DFlash / 1 | 2 | 0 | 2 | 0 |
| DFlash / 0 | 1 | 0 | 0 | 0 |
| DFlash2 / 1 | 2 | 0 | 2 | 0 |
| DFlash2 / 0 | 1 | 0 | 0 | 0 |
| DSpark / 1 | 2 | 2 | 2 | 2 |
| DSpark / 0 | 1 | 1 | 0 | 0 |

计数里的 `objective_complete` 在 backward 可能不出现，因为 non-reentrant checkpoint 可以在恢复必要中间量后提前停止重算；这不是 objective 没有被进入。日志中的 `softmax/grad` 只表示调用时的 grad mode，不表示 detached teacher 获得了梯度。

## 6. 历史性能记录：不能用于最终版本验收

原始文件：[loss_fusion_fp32_results.json](loss_fusion_fp32_results.json)。文件已标注 `contended_gpu_not_for_speedup_claims`，并注明早于 L1 原生 softmax 修正。

当时配置：FP32 logits，shape `[512, 151936]`，每轮 30 次前向+反向，7 轮；H800；CE+L1 测量 objective 为 `0.1 * CE + L1`。表中耗时为轮次中位数；显存为进程 `max_memory_allocated - baseline` 的额外峰值，不是整卡占用或完整训练峰值。

| 路径 | eager 历史 ms | compiled 历史 ms | eager 额外峰值 MiB | compiled 额外峰值 MiB |
| --- | ---: | ---: | ---: | ---: |
| CE | 0.974 | 0.515 | 890.25 | 296.75 |
| CE+L1 | 4.109 | 1.236 | 1483.76 | 296.77 |
| KL | 2.662 | 1.679 | 1483.75 | 593.51 |

**不要据此计算最终加速比或宣称这些显存数字属于当前实现。** 计时受其他进程干扰，且最终 L1 路径需要保留原生 softmax 概率张量。原始轮次里有明显波动，例如 CE eager 从约 0.96 ms 到 2.14 ms。AB/BA 交替和多次取中位数无法自动消除这种系统性干扰。

该脚本仅测 loss；不包含 lm_head GEMM、attention、SGLang teacher、数据加载、FSDP/Muon 通信、optimizer 或真实端到端 step。FP32→BF16 临时张量清理和投影复用的端到端收益也尚无可靠新测量。

## 7. 新机器复现步骤

以下命令从迁移后的仓库根目录执行，假定已配置好 `.venv`。所有实验配置、日志、分析文件放在 `tests/`。`CUDA_VISIBLE_DEVICES` 需按实际空闲 GPU 设置。

先记录状态和环境：

```bash
mkdir -p tests/optimization_handoff_runs
git rev-parse HEAD > tests/optimization_handoff_runs/head.txt
git status --short > tests/optimization_handoff_runs/worktree.txt
git diff --binary > tests/optimization_handoff_runs/worktree.patch
nvidia-smi > tests/optimization_handoff_runs/nvidia-smi.txt
.venv/bin/python -m torch.utils.collect_env > tests/optimization_handoff_runs/environment.txt
```

这里的 patch 不包含未跟踪文件，不能替代完整源码迁移。性能测量期间持续核实 GPU 独占/负载情况；一张测前快照并不能证明整个实验期间没有干扰。

### 先跑正确性

```bash
export PYTHONPATH=.
export OMP_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$PWD/tests/.cache/loss-fusion"

.venv/bin/python -m pytest tests/test_loss_fusion.py -q \
  > tests/optimization_handoff_runs/loss-tests.log 2>&1

.venv/bin/python -m pytest \
  tests/test_training_compute_reuse.py \
  tests/test_training_refactor.py \
  tests/test_dflash_family.py -q \
  > tests/optimization_handoff_runs/framework-tests.log 2>&1

LITESPEC_LOSS_BACKEND=torch .venv/bin/python tests/inspect_objective_recompute.py \
  > tests/optimization_handoff_runs/recompute.log 2>&1

CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --standalone --nproc_per_node=2 \
  tests/distributed_loss_fusion_smoke.py \
  > tests/optimization_handoff_runs/fsdp-loss.log 2>&1

CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --standalone --nproc_per_node=2 \
  tests/distributed_muon_smoke.py \
  > tests/optimization_handoff_runs/muon.log 2>&1
```

检查各命令退出码及日志中的 PASS/pytest summary。CUDA 测试在无 GPU 时会跳过，不要把 skips 解释为 CUDA 已验证。分布式 loss smoke 自行使用按 rank 分开的 `tests/.cache/distributed-loss/<rank>` 编译缓存。

### 再跑最终实现的 loss 微基准

确认空闲后分别测 FP32、BF16，使用新文件名，不覆盖旧的受干扰记录：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tests/benchmark_loss_fusion.py \
  --rows 512 --vocab 151936 --dtype float32 --iterations 30 --trials 7 \
  --note "Record actual GPU occupancy conditions here" \
  --output tests/optimization_handoff_runs/loss_fp32.json \
  > tests/optimization_handoff_runs/loss_fp32.log 2>&1

CUDA_VISIBLE_DEVICES=0 .venv/bin/python tests/benchmark_loss_fusion.py \
  --rows 512 --vocab 151936 --dtype bfloat16 --iterations 30 --trials 7 \
  --note "Record actual GPU occupancy conditions here" \
  --output tests/optimization_handoff_runs/loss_bf16.json \
  > tests/optimization_handoff_runs/loss_bf16.log 2>&1
```

脚本先为每条路径预热 3 次，再用 CUDA event 计时，按轮次交替 eager/compiled 顺序；输出每轮时间、中位数和额外 allocated 显存峰值。仍需确认没有新 shape 重编译进入计时区间。冷启动编译耗时应单独记录。

继续覆盖真实 active token 数和词表大小，例如 rows=128/512/2048、vocab=32768/151936；大 shape 先确认显存足够。`inductor` 标签表示请求该 backend，不代表 BF16 CE 被强行融合。BCE 尚未包含在此微基准的计时项目中，只有正确性覆盖。

### 最后测真实训练

固定初始化/checkpoint、样本顺序、seed、batch、accumulation、anchor、objective chunk 和日志频率，分开启动 `LITESPEC_LOSS_BACKEND=torch` 与 `auto` 两组。需要研究门槛时再加入 `inductor`。两次运行使用独立输出目录；不要同时争用同一组 GPU 做 A/B。

YAML 启动入口示例（先把配置复制到 `tests/` 并改好路径、节点数和输出目录）：

```bash
LITESPEC_LOSS_BACKEND=torch .venv/bin/python scripts/config/launch_dlite.py \
  sft tests/experiment_dlite_torch.yaml

LITESPEC_LOSS_BACKEND=auto .venv/bin/python scripts/config/launch_dlite.py \
  sft tests/experiment_dlite_auto.yaml

LITESPEC_LOSS_BACKEND=auto .venv/bin/python scripts/config/launch_dflash_family.py \
  dspark tests/experiment_dspark_auto.yaml
```

上面的实验 YAML 是待创建文件，不是本次交接已生成的配置。DFlash/DFlash2 使用对应 algorithm 和配置。多节点需按 launcher 的要求为各节点提供正确 node rank/master 地址，不能沿用所有节点均为 rank 0 或跨机器使用 `127.0.0.1`。

建议按以下顺序验证：

1. 单卡或单节点短跑检查 loss、梯度有限性、有效 token denominator、更新步数和恢复行为，再逐步扩到目标分布式规模。
2. 排除预热/编译后，分别记录普通 step 与日志 step；报告 step 时间分布、有效 tokens/s、每 rank 的 allocated/reserved 峰值和最慢 rank。多卡不要仅看 rank 0 的局部耗时。
3. 分别比较 `objective_chunk_blocks=0` 和实际分块配置，这是独立的显存/重算实验，不能与 backend 改动混在一个 A/B 中。
4. 用 profiler 检查 draft/teacher 投影次数、loss kernel、分配和通信。profiling 结果与无 profiler 的吞吐测量分别保存。
5. 继续运行超出此前约 400 step 的故障区间，保存报错 rank 的最早 traceback。小规模 smoke 不能替代该验证。

如要测“所有优化相对原始代码”的总收益，应另外建立经过核对的优化前 baseline；`0e65148` 已含部分优化，当前 `afbb9ed` 则包含最终修改，不能把前者简单命名成所有优化之前的 baseline。

## 8. 后续修改注意事项

- 保留用户的“不分桶、不加 DSpark teacher 缓存”约束。
- 不更改 loss label、权重/decay、分母、teacher detach、selector mask 或梯度累积的含义来获得表面加速。
- 编译器版本/GPU 变化后重跑数值检查，特别是相等及近似相等分布下的 L1 梯度、低精度 CE、mask 和混合精度路径。
- 如继续实现 fused AdamW，单独比较参数更新、optimizer state、scheduler 和恢复；不要把已有 `copy_` 优化等同于 fused AdamW。
- DSpark 仍计算其 objective 中的诊断项。若进一步减少非日志计算，先明确哪些结果仅用于日志，哪些同时是 confidence 目标或训练 loss 的输入。
- 如编译失败，可先用 `LITESPEC_LOSS_BACKEND=torch` 做对照定位；当前没有对所有编译异常自动吞错回退的机制。
- 接收机器 `223.76.225.211:56780` 此前从当前环境探测返回 TCP `Connection refused`，未进入 SSH 密钥验证阶段；连接和认证均未完成。该记录只代表当时的探测结果，不是持续监控结论。

交接完成的标准：新机器能复现正确性检查，并为最终代码生成可信的独占 GPU 微基准和真实训练测量；在此之前，不承诺吞吐提升比例或长程训练完全一致。


## 9. 新机器接续验证记录（CUDA 13.0）

2026-10-13 在 `/diancpfs/user/wanghanzhen/LiteSpec` 接续。当前 HEAD 为
`97e65478e6425cd81915b34b50d53914bc3e804a`，已确认 `afbb9ed` 是其祖先；
两者之间没有公共 loss/core 的变化。已有 CUDA 12.8/13.0 安装配置修改继续保留。

当前环境为 Python 3.11.17、PyTorch 2.9.1+cu130、CUDA runtime/toolkit 13.0，
8 张 NVIDIA L20Z（计算能力 9.0）。目标模型为
`/diancpfs/user/wanghanzhen/artifacts/models/Qwen3-8B`。该模型实际有 **36 层**、
hidden size 4096、vocab size 151936；final-norm ID 是 **35**，不能由
`target_layer_ids` 中的 31 推断最后一层。

本次结果目录：`tests/optimization_handoff_runs/20261013_cuda13_203502`。

- `loss-tests.log`：31 passed，重新验证了相等/近似相等分布的梯度、混合精度和 checkpoint 边界。
- `framework-tests.log`：47 passed，覆盖计算复用、训练重构和 DFlash 系列。
- `recompute.log`：当前 CPU 重算计数与梯度对照通过。
- `fsdp-loss.log`：双卡 FSDP、梯度累积和恢复 PASS，观测 max master error=0。
- `muon.log`：matrices_per_rank=1、2 均 PASS，观测 max master error=0。

### 新微基准

`verified_microbench/` 覆盖 rows=128/512/2048、vocab=32768/151936 和
FP32/BF16，共 12 组，30 iterations × 7 trials。第一次未加入 PID 映射的
测量保留在结果根目录，汇总使用 `verified_microbench/` 的复测结果。
新脚本记录首次调用和三次预热的 wall time，并断言计时轮次没有新增编译图。

为处理容器与驱动 PID 不一致，先观察 CUDA 上下文建立前 GPU 没有计算进程，
再将分配后唯一的驱动 PID 与本次 benchmark PID 对应。测量期间持续采样，
核对同一 GPU UUID 上没有其他可见进程。采样实际最大间隔约 2.43 秒，
不排除采样间隔内的短暂外部活动。原始记录见 `gpu-monitor.jsonl`；
每组核对结果、原始时间和显存记录见 `microbench_summary.json`。

shape [512,151936] 的 FP32 CE/CE+L1/KL，eager→inductor 中位数分别为
0.964→0.454 ms、3.953→2.888 ms、2.661→1.643 ms。
BF16 CE 仍为原生路径，0.513→0.515 ms；BF16 CE+L1 为
3.974→3.463 ms，KL 为 3.297→1.705 ms。
最终 CE+L1 的额外 allocated 峰值基本不变，不能沿用旧版本的显存下降结论。

这些数据只覆盖 loss 前向+反向，不能解释为真实训练 step 加速比。
小 shape（尤其 [128,32768]）存在明显编译回退收益不足，当前 auto 门槛仍待校准；
部分 BF16 CE 原生对照也有波动，应加长测量后再设新阈值。
公共 loss 数值保护保持原样，尚未因性能结果修改生产路径。

### 正式数据和 full cache

用户指定输入：`cache/data/prompts/wildchat_chinese_turns_le2.jsonl`，
4,586 条记录、5,794 个用户回合。已验证物理 JSONL 行能正确解析；
不能用 `str.splitlines()` 拆分这份输入，因为用户文本包含 Unicode 行分隔符。

按照用户选择保存 0、8、16、24、31 层，并另外自动保存第 35 层 final-norm；
max_length=10240，TP=1，batch=1，shard_size=512。先以 Qwen3-8B
生成 assistant 回答（temperature=0.7、top-k=15、max_tokens=4096、thinking 关闭），
恢复输入记录顺序后再生成 `regen_full`。5 条 probe 的生成、缓存、训练读取器
checksum 和 tensor shape 检查通过。全量阶段状态和命令保存在
`cache_pipeline/state.json`；打包已恢复完成，4,586 条成功、0 错误、9 个分片，
共 298,266,081,280 bytes（约 277.8 GiB）。新机器的完整读取器校验通过：所有分片 SHA-256 匹配；
首尾及分片边界的 6 个样本 tensor 检查通过。block_size=8 时 4,568 条 eligible，
18 条由现有规则判为不符合条件，仍完整保存在缓存中。

预定输出：

- token-only：`cache/data/regen_token_only/wildchat_chinese_turns_le2_qwen3_8b_temp0.7_topk15_max4096.jsonl`；
- full cache：`cache/data/regen_full/wildchat_chinese_turns_le2_qwen3_8b`。

先前在线 fixture 训练启动被 SSH 断开导致的 SIGHUP 中断，未据此得出训练失败或数值结论。
后续数据 pipeline 使用独立 session，子任务也使用 `start_new_session=True`，
避免 SSH 断开终止 torchrun。用户已要求先完成正式 full cache，后续真实训练
对照将使用该缓存；offline 吞吐不能与旧 online/48-rank 配置直接比较。

### 更换服务器并恢复打包（2026-10-13）

用户将 SSH 地址改为 `root@47.237.165.176 -p 971`，并确认旧服务器已经停止或释放。
两台服务器共享 `/diancpfs`。新机器有 2 张 NVIDIA L20Z；原 `.venv` 指向旧机器
的本地 `/root/.cache/uv`，已在新机器本地按相同锁文件重建，未更换共享 `.venv` 链接。
PyTorch 2.9.1+cu130、BF16、causal-conv1d 前向/反向、SGLang Engine、
sgl-kernel 和 flashinfer rmsnorm 检查通过，`uv pip check` 通过。
安装和验证记录保存在 `tests/environment-setup/newhost-uv-sync.log`。

全量回答生成已经完成：4,586 条记录，0 错误。原隐藏状态生成任务在最后打包阶段
中断，已有 8 个完整分片；暂存文件与已提交分片的样本索引合起来完整覆盖
0..4585，错误日志为空。使用仓库 `finalize_staging` 恢复最后 490 条的打包，
不重新生成回答或隐藏状态。恢复状态见 `cache_pipeline/recovery_state.json`；
最终读取器校验结果将写入 `cache_pipeline/full_validation.json`。

后续正式数据 smoke 和吞吐/稳定性对照采用单机 2 rank、TP=1 的 offline DLite。
保持 batch=1、accumulation=1、seed=42、num_anchors=512、block_size=8 和
原 loss/Muon 设置。结果放在 `newhost_2gpu/`；不能据此宣称旧 48-rank online
配置约 400 步故障已修复。

### 正式缓存双卡训练验证结果（2026-10-14）

结果目录：`newhost_2gpu/`，总状态 `completion.json`。full cache 的 9 个分片
全部通过正式读取器 SHA-256 检查；来源元数据、输入顺序和 5,794 个回答回合
核对通过。完整记录分别见 `cache_pipeline/full_validation.json` 和
`cache_pipeline/generated_metadata_validation.json`。

为减少同一不可变缓存的重复全量扫描，实验工具增加可选
`--verified-cache-record`：复用一次完整读取器 checksum 审计，同时每次启动
都核对 manifest SHA-256 和分片 inode、device、size、mtime、ctime；任一
变化即停止。仍执行读取器的层号、模型、索引和形状检查。此复用位于 `tests/`
诊断工具中，不是 target logits/probabilities 的缓存，也没有修改训练数学路径。

3 步 smoke 通过。第一版诊断工具把顺序头 `project_logits` 当成了外层模型接口，
在正式训练开始前报 AttributeError；已修正为 draft sequential head，并且只在
独立 profiling pass 安装投影计数器。失败记录保留于 `smoke_auto.log`，通过的
记录为 `smoke_auto_v2.log`。

torch → auto → auto → torch 四轮各 60 步，对应 `torch_a/`、`auto_b/`、
`auto_b_repeat/`、`torch_a_repeat/`。两 rank 的初始化 FP32 master hash、
每步输入和 anchor 指纹均一致；60 步全局加权 loss 的最大绝对差约 1.22e-4。
共同排除前 5 步预热、产生新图的第 8 步和第 50 步日志后，采用相同的 53 个
正常步骤。最慢 rank 的中位 step：torch 239.73/239.86 ms，auto
233.17/232.86 ms；初步全局输入吞吐 torch 9635/9648 tokens/s，auto
9868/9865 tokens/s。此约 2.3% 的初步差异不满足后续明确的 200 步预热要求，
不作为正式吞吐结论。第 50 步日志：torch 238.36/240.25 ms，
auto 230.69/231.76 ms。allocated 峰值 torch 28.61 GiB，auto 24.55 GiB；
reserved 峰值 torch 35.32/35.37 GiB，auto 32.81 GiB。计时包括 batch 获取
和输入 fingerprint，不能解释为单独 loss 的加速比。原始记录和汇总见
`abba_summary.json`。

持续 NVML 采样与两 rank 的 GPU UUID、驱动 PID 对齐，稳定区间采样中未观察
到额外计算进程，实际最大采样间隔约 1.11 秒；不能排除更短暂的活动。记录见
`gpu_isolation_audit.json`。

独立第 10 步 CPU profile 没有新增编译图，draft logits 投影每 rank 为 1 次。
当前 PyTorch 2.9.1+cu130/Kineto 的 trace 只有 CPU 事件，最小 CUDA matmul
probe 也报告 GPU trace empty，因此未据此报告 CUDA 算子时间。改用机器上
Nsight Systems 2025.3.1，以 cudaProfilerApi 仅捕获第 10 步，得到两张 GPU
共 4,371 个 kernel 事件；该 pass 同样没有新增编译图，投影计数为 1。
实际 GPU 报告和统计为 `nsys_profile_step10.nsys-rep`、
`nsys_profile_step10_stats.log`，元信息见 `nsys_state.json`。

450 步 auto 稳定性验证和最终 checkpoint 保存通过。两 rank 每步 loss、
grad norm 均为有限值，最后 loss=2.23256，最大 grad norm=3.05，最高
allocated=24.64 GiB、reserved=32.81 GiB；新增编译图只出现在 1、2、8 步。
结果见 `stability_summary.json`，验证终点 checkpoint 在
`stability_auto/final/`。这只是 450 步的验证终点，并不是完整 6 epoch 训练；
其 LR schedule 保留 6 epoch（13,704 optimizer steps）。

这些结果只验证当前单机 2-rank offline DLite SFT；不证明旧 48-rank online
配置约 400 步故障已经修复。auto 的小 shape 门槛校准、其他算法实际训练和
原多节点 online 故障仍需在对应配置上验证。

### 吞吐复测：每轮预热 200 步（2026-10-14）

结果目录：`newhost_2gpu/warmup200_steps300/`。按用户要求，torch → auto → auto → torch
四轮各运行 300 步，前 200 步全部排除；只统计共同的第 201–300 步，
正常步骤 98 个（排除日志步骤 250、300）。这一区间四轮均没有新增编译图。
每 rank 的初始 FP32 master、全部 300 步输入和 anchor 指纹一致。
单机 2 张 L20Z，offline DLite SFT，batch=1、accumulation=1、seed=42、
TP=1、num_anchors=512、block_size=8，LR schedule 仍为 6 epoch / 13,704 步。
本次吞吐定义为共同窗口内全局输入 token 总数 / 最慢 rank step 耗时总和，
包含 batch 获取和输入 fingerprint；日志、profiling 和前 200 步单列。

- 第一次配对：torch 正常 step 中位数 240.53 ms，输入吞吐 10,673.61 tokens/s；
  auto 234.38 ms、10,715.46 tokens/s，吞吐差异 +0.39%。
- 反向顺序复测：torch 240.82 ms、10,666.13 tokens/s；
  auto 233.59 ms、10,917.89 tokens/s，吞吐差异 +2.36%。
- 合并两轮 token 和耗时：torch 10669.87 tokens/s，
  auto 10815.73 tokens/s，观察到 +1.37%。
  各次吞吐差异为 0.39%–2.36%，不再采用旧 60 步测试的 +2.3% 作为正式结论。
  step 中位数下降约 2.6%–3.0%；中位数与按总耗时计算的吞吐是不同指标。
- 250/300 日志步骤：torch 243.77/237.94 ms 和 243.01/238.88 ms；
  auto 234.49/231.72 ms 和 238.59/231.45 ms。
- 整轮 allocated 峰值 torch 28.70 GiB、auto 24.64 GiB；
  reserved 峰值 torch 35.37 GiB、auto 32.81 GiB。
- 计入所有符合条件的窗口耗时，没有删掉无新增编译图的慢步骤。auto 第一次
  第 210/296/221 步为 550.39/487.92/350.77 ms，反向顺序复测也有慢步骤；
  因此不能只用 step 中位数推导统一吞吐加速比。
- 稳定窗口 NVML 审计通过，没有观察到额外 GPU 计算进程，最大采样间隔
  1.11 秒；采样仍不能排除更短的外部活动。

数值记录需要区分初始化/采样复现和训练轨迹：同 backend 的重复运行在全部
300 步记录上 weighted loss 完全一致；torch 与 auto 最大绝对差 0.015278
（第 281 步），第 300 步差 0.002899。未出现非有限 loss 或 grad norm，
但不能据此声称跨 backend 长期训练轨迹完全等价，此为修复前记录；差异在后续小节中已定位并修复。

新正式汇总为 `abba_summary.json`，含 warmup_steps=200、
measurement_step_range=[201,300]、共同步骤索引、tokens/s、日志步骤、
完整显存峰值及数值差异。隔离检查见 `gpu_isolation_audit.json`；
每步原始记录在各 trial 的 `telemetry/steps_rank_0.jsonl` 和
`steps_rank_1.jsonl`，启动配置及日志也保存在该目录。

### 稳定训练分阶段计时、CUDA Graph 与数值差异定位（2026-10-14）

本轮已完成六环节计时、稳定期 Nsight kernel 审计、真实 loss 输入及完整模型反事实实验，并修复 CE 编译导致的 BF16 梯度舍入差异。所有脚本、原始数据和报告均保存在 `tests/`；没有 commit/push。

**配置及计时方法。** 单机 2×L20Z，Torch 2.9.1+cu130，Qwen3-8B full hidden cache；训练选取 hidden 0/8/16/24/31，final-norm 35 用于冻结 target head 投影。保持 batch=1、accumulation=1、512 anchors、block=8、5 层 draft、RNN rank=320、CE=0.1/L1=1、Muon/AdamW、seed=42、原采样顺序和 13,704 步 LR schedule。每轮先运行 200 步，再统计 201–300 的共同正常步骤，排除日志 250/300；没有删除其他慢步骤。

六段采用同一 CUDA stream 上的共享边界 event，不在各段之间 synchronize；它们是训练关键时间线的归属（包括 CPU 导致的空闲、依赖和通信等待），不是纯 kernel 运算时间。CPU 主线程跨度另存。每步选整步较慢的 rank，再汇总该 rank 六段，因此同一份占比能相加为 100%。主线程 `next(data)` 暴露等待计入加载数据，后台 worker 读取/解包与此前计算重叠，不能把它的耗时再加到整步上。

准备段包括 H2D、anchor/hidden gather、冻结 target LM-head 从缓存 final-norm 投影 logits、draft prepare_batch 和进入 backbone 前的 FSDP all-gather。forward 包括 backbone、RNN 和 draft logits 投影；后两者虽写在 compute_supervised_loss 内，已单独归入 forward。loss 包括 CE/L1、mask/权重/归一化和当前每步指标。backward 包括反向通信以及发生于反向中的重算；update 包括梯度归一化、norm/clip、Muon/AdamW、scheduler 和 copy-back。

**修复前分段基线。** `phase_diagnosis_20261014/phase_summary.json`：auto 95 个样本（额外排除 Nsight 的 240–242），torch 98 个样本。auto 平均 wall=236.03 ms、六段合计=235.66 ms，额外 fingerprint=0.278 ms、边界及最终同步余量=0.091 ms。阶段钩子的初始化、全部 300 步 input/anchor、loss 和 grad norm 与之前各自 backend 的基线完全一致。

- 加载数据：auto 0.448 ms / 0.19%；torch 0.310 ms / 0.13%。
- forward 前准备：auto 13.934 ms / 5.91%；torch 13.557 ms / 5.63%。
- forward：auto 23.278 ms / 9.88%；torch 22.748 ms / 9.46%。
- loss 计算及指标：auto 21.238 ms / 9.01%；torch 24.785 ms / 10.30%。
- backward：auto 68.676 ms / 29.14%；torch 71.238 ms / 29.61%。
- 参数更新：auto 108.086 ms / 45.86%；torch 107.945 ms / 44.87%。

**CUDA kernel 与 Graph 实验。** `phase_diagnosis_20261014/stable_240_242.nsys-rep` 仅捕获预热后的 240–242 步；两卡共 13,179 个 kernel，约 2,196.5 个/卡/步。5,112 个（38.79%）低于 10µs，6,631 个（50.31%）低于 20µs。低于 10µs 的 kernel 只占跨卡/stream 累计 kernel 时长 1.11%，此累计量不能当整步 wall time。CUDA Graph API 次数和非零 graphNodeId 都为 0。

按 launch 所属 CPU NVTX range + process/correlationId 归属（包括 autograd 工作线程），update 4,472 个 kernel、backward 4,908、forward 2,665、loss 762、prepare 348；仅 24 个在六段之外。update 中有大量 dtype cast/copy、逐参数 scale、norm、momentum/weight decay 等；较重的 Muon Newton-Schulz GEMM 和 NCCL SendRecv 也在 update。因此更新约 46% 并不等于这 46% 全部可以通过减少小 kernel 消除。

当前完整 step 有动态 active-position 形状、数据加载、CPU 分支，以及 `normalize_accumulated_gradients` 的 denominator.item、BF16Optimizer 的 active-mask.tolist / grad_norm.item；不能直接捕获整步。优先候选是静态形状 Newton-Schulz 子过程、固定 buffer 的参数 cast/scale/copy，以及可保持原梯度舍入边界的 foreach/pointwise 融合。CUDA Graph 是捕获并批量重放 launch，不等同把多个数学 kernel 融成一个。

`phase_diagnosis_20261014/muon_cudagraph_probe.json` 已实际验证纯 CUDA Newton-Schulz 可 capture/replay：每种 shape 200 次调用预热、eager/graph/graph/eager 各 100 次；输出逐元素完全一致。640×640：约 0.271→0.096 ms/次（约 64%）；4096×4096：3.544→3.357 ms（约 5.3%）；151936×320：1.904→1.867 ms（约 1.9%）。这是子过程 microbenchmark，尚未接入正式 optimizer，不声称整步获得上述加速。大矩阵计算和通信仍占主要时间。

**数值差异的因果定位。** 修复前相同初始化、300 步 input/anchor；同 backend 重复 loss 完全一致，但跨 backend 第 1 步 loss 相同、grad norm 已相差约 3.0e-6，第 2 步 loss 开始分叉，最大到 0.015278。

真实第 1 步 rank 0 的 loss 输入 shape=[1127,151936]，draft logits 是 BF16 head 输出后 cast 成 FP32，teacher 为 BF16。`numeric_diagnosis_20261014/torch/numeric/component_probe.json`：单独 L1 的 FP32 梯度完全一致；compiled CE 的 FP32 梯度最大差 4.55e-12，8,659,927 元素不同，其中 1,316 个在转为 BF16 后仍不同。CE+L1 组合仍有 924 个 BF16 梯度不同。原保护只检查 loss 接口 logits.dtype，因此这条已 cast 为 FP32 的真实训练路径仍进入 compiled CE。

进一步的 `ce_boundary_probe.json` 使用同一真实输入的 32 行定位到 log-softmax：native log-softmax + native NLL 与 CE 基准一致；compiled log-softmax + native NLL 复现 compiled CE 的梯度差异；native log-softmax + compiled gather/reduction 的梯度完全一致。compiled log-softmax 的输出最大差 1.91e-6。Inductor 生成源码展示分段 max/sum 归约和 Triton log-softmax，这条不同的浮点实现导致 FP32 梯度落在 BF16 舍入边界两侧，进而改变网络反向及后续优化轨迹。不能因为 scalar loss close 就认为 BF16 参数更新相同。

完整网络的 first-step snapshots 见 `numeric_diagnosis_20261014/snapshot_comparison.json`：旧 auto 在两 rank 分别有 44,735,412 / 16,771,675 个 BF16 model-gradient 元素不同；FP32 master 更新后分别有 239,639,681 / 212,126,267 个元素不同，其中最终转 BF16 后仍有 36,382 / 25,053 个不同。只恢复 native CE、保持 compiled L1 的反事实实验，两 rank 全部 model gradients 和全部 FP32 master 更新均逐元素一致。

**修复和复测。** `litespec/core/losses.py` 保留所有 dtype 的 native CE forward/backward，继续编译 native softmax 后的 L1 reduction；public weighted_cross_entropy 同样保护 FP32 来自 BF16 的梯度边界。没有修改 loss 数学公式、数据、anchors 或 optimizer。`tests/test_loss_fusion.py` 新增真实词表尺寸、BF16 head→FP32 loss 的 CE/CE+L1 精确梯度回归，两例在修复前失败、修复后通过；loss suite 33 passed；projection/recompute、Stage-1 loss、runner 相关测试 74 passed，记录见 `related_training_tests_gpu_available.log`。

修复后 `fixed_ce_warmup200_steps300/summary.json`：torch→auto→auto→torch 四轮各 300 步，前 200 步全部预热，共同正常样本 98 个/轮。两 rank 四轮 initial master、input 和 anchors 相同；全部 300 步 grad norm 完全一致，最终全部 FP32 master 的 SHA-256 四轮完全一致。跨 backend weighted loss 最大差 2.591307213e-07，来自 compiled L1 求和的浮点舍入，已不产生参数轨迹分叉。

**当前修复后的正式六环节占比**（合并两轮 auto，共 196 个稳定样本）：

- 加载数据：auto 2.882 ms / 1.19%；torch 0.204 ms / 0.08%。
- forward 前准备：auto 14.816 ms / 6.13%；torch 13.891 ms / 5.75%。
- forward：auto 23.142 ms / 9.57%；torch 23.037 ms / 9.54%。
- loss 计算及指标：auto 22.862 ms / 9.46%；torch 24.682 ms / 10.22%。
- backward：auto 70.093 ms / 28.99%；torch 71.707 ms / 29.70%。
- 参数更新：auto 107.966 ms / 44.66%；torch 107.891 ms / 44.69%。

数据加载的暴露等待有长尾：两轮 auto 的中位数约 0.155–0.158 ms，但第一次在 step 210/221 为 300.89/166.95 ms，第二次 step 210 为 64.09 ms。上述均不是新编译步骤，全部保留在均值、占比和吞吐中；没有用删除慢步骤的方法提高加速比。分布及原始位置见 summary.json 的 data_wait_distribution。准备阶段均值也包含 CPU 暴露等待与通信依赖，纯 CUDA kernel 时间另见 Nsight。

合并稳定窗口输入吞吐：torch 10666.39、auto 10650.63 tokens/s，观察差异 -0.15%。当前复测没有显示可靠的整体吞吐收益，不能沿用修复前 compiled CE 的性能数字。

- torch_a：median 240.966 ms，mean 241.379 ms，input 10683.57 tokens/s，allocated 峰值 28.698 GiB、reserved 35.371 GiB。
- auto_b：median 237.794 ms，mean 244.343 ms，input 10553.94 tokens/s，allocated 峰值 26.669 GiB、reserved 32.809 GiB。
- auto_b_repeat：median 237.617 ms，mean 239.907 ms，input 10749.10 tokens/s，allocated 峰值 26.669 GiB、reserved 32.809 GiB。
- torch_a_repeat：median 241.199 ms，mean 242.156 ms，input 10649.26 tokens/s，allocated 峰值 28.698 GiB、reserved 35.371 GiB。

稳定窗口 GPU UUID/driver PID 连续采样审计通过，见 `fixed_ce_warmup200_steps300/gpu_isolation_audit.json`；采样无法排除短于间隔的外部活动。本轮证明当前双卡、指定 hidden 的 offline DLite SFT；没有据此扩大到其他算法或旧多节点 online 故障。

脚本入口：`tests/benchmark_training_phases.py`、`training_phase_profiler.py`、`benchmark_training_numeric.py`、`training_numeric_probe.py`、`probe_ce_boundary.py`、`probe_muon_cudagraph.py`；控制器 `tests/environment-setup/run_phase_diagnosis.py`、`run_numeric_diagnosis.py`、`run_fixed_ce_abba.py`。各 config、init/step/phase/complete telemetry、Nsight SQLite 和测试日志均保留。


### 当前保留的优化与速度收益清单（2026-10-14）

本清单记录只日志指标功能加入前的优化、验证和数值修复；后续新增功能见本文末尾。当前 torch/auto ABBA 只隔离公共 loss backend；两组都启用 head 复用、参数 copy 等已有优化。因此它不是“所有优化相对于原始框架”的总对照，也不能累加各局部加速比。

1. **DLite transition 共用 sequential head/logits。** Stage-1 KL 与 supervised objective 共用同一次 student RNN/head 投影，调用由两次变为一次，梯度保持在共享计算图中。计算复用测试通过；没有 transition 优化前后的正式 step 耗时。当前 SFT 吞吐测试没有进入 transition，不能将该收益归入当前 SFT。
2. **DFlash/DFlash2 日志复用 draft logits。** 日志指标使用 objective 已生成的 logits，删除日志时额外的 draft 词表投影。带指标的原始前向每 chunk 两次投影变为一次；未独立量化真实训练加速。
3. **checkpoint 重算跳过日志回调。** 日志只在原始前向执行，backward replay 不再重算该回调里的投影/指标（包括仅用于日志的 teacher 投影）。训练 objective 需要的重算仍保留；CPU 重算计数和梯度对照通过，未独立量化 GPU step 加速。
4. **DSpark 诊断复用。** 接受概率、距离、teacher/student top-1 概率复用 L1 路径结果，减少诊断重复 softmax/归约。confidence 的训练目标及真实 objective 中的 teacher 投影保持；未测独立耗时收益，没有 teacher projection cache。
5. **FP32 master→BF16 model 直接 copy。** update 和恢复改为目标参数 copy_(master)，替代先转换出完整 BF16 临时张量再 copy。减少临时分配与搬运；正确性和恢复验证通过，尚无独立计时。FP32 master、状态和 FP32 梯度仍保留。
6. **L1 归约编译融合。** 保留原生 teacher/student FP32 softmax，编译后续差值、abs、加权和归约；当前 CE 全部原生。双卡 full-cache SFT、每轮预热 200 步的最新观察：loss 阶段 24.682→22.862 ms（耗时 -7.37%）；整个 backward 71.707→70.093 ms（-2.25%）。这些是训练阶段时间线，含指标/等待/通信，不是单独 L1 kernel 计时。allocated 峰值 28.698→26.669 GiB（少 2.029 GiB，约 7.07%）。
7. **KL 编译及 frozen-teacher 边界。** teacher log-prob/prob 在编译 student 路径前计算并 detach，避免 AOT 自身 backward 重新算 teacher normalization。此前独占 GPU 的 [512,151936] 算子 forward+backward 微基准：FP32 2.661→1.643 ms（耗时 -38.3%），BF16 3.297→1.705 ms（-48.3%）。KL 代码在本轮 CE 修复中未改；这些是历史算子微基准，不是预热 200 步后的 KL 真实训练吞吐。
8. **BCE 适用路径编译。** FP32 target 时允许编译 confidence BCE，CPU/小 shape 的 auto 路径仍 eager。正确性覆盖通过；没有 BCE 独立速度记录。
9. **公共 loss 大小派发。** auto 只在 CUDA 且元素数达到 65536 时启用适用编译，小输入保留 eager，减少小 workload 的编译/派发成本。该门槛尚未根据可靠逐 shape 结果校准，不声称已有统一加速。
10. **启用指定层 full hidden cache 训练。** 本次生成并使用现有缓存机制，保存指定 5 层和 final-norm；训练读取 target hidden，跳过在线 target Transformer，但仍计算 target LM-head logits。未在相同硬件/配置下做 online/offline A/B，因此不提供加速倍数。这是启用已有训练模式，不是本轮新增的训练算法。

仓库已有、这次验证的底层优化包括 **FSDP sharded Muon 的按 shape 分组、矩阵批处理、async all-to-all 与下一组预取重叠**，以及 **foreach grad norm + 单个标量的 norm all-reduce**。本轮验证 matrices_per_rank=1/2 的正确性，但没有重新做各项性能消融；不能把当前约 108 ms 的 update 当作这些优化的节省量。

兼容性和测试启动改进不算训练加速：小 flex mask 的 eager 回退修正 sparse-count 错误；CUDA 12.8/13.0 可选环境及本机编译缓存隔离；测试启动复用已经完成的 277.8 GiB 缓存 checksum 审计，并验证文件 stats 与 manifest SHA。上述没有改变单步训练数学。

当前整步结果：四轮 300 步，先预热 200 步，共同正常窗口各 98 步；torch→auto 两组整步中位耗时分别 240.966→237.794 ms 和 241.199→237.617 ms（耗时 -1.32% / -1.48%）。按全部正常窗口 token / 耗时合并，吞吐 10666.39→10650.63 tokens/s（-0.15%，本次基本持平）；两次配对分别 -1.21% 与 +0.94%。数据预取等待长尾全部保留。最终 FP32 master 在两 rank、四轮均完全一致；这些已量化的收益只归属于本配置中的公共 loss backend 对照。

**已撤回或尚未接入：**

- FP32 CE/log-softmax 编译已因 BF16 梯度舍入导致参数轨迹分叉而撤回；旧 CE 加速和修复前整体 +1.37% 不计入当前收益。
- 把 L1 两侧 softmax 一起编译的实现已撤回；当前保持原生 softmax，保护相等/近似相等分布的梯度。
- CUDA Graph 只完成静态 Muon Newton-Schulz 子过程实验，每种 shape 200 次预热：640×640 0.271→0.096 ms（耗时约 -64%）、4096×4096 3.544→3.357 ms（约 -5.3%）、151936×320 1.904→1.867 ms（约 -1.9%），输出完全一致。尚未接入正式 optimizer，正式训练收益为未验证。
- AdamW 没有启用 fused=True；grad scale/clip/copy 尚未全面融合。本条记录之后已实现非日志窗口跳过诊断指标，详见后续小节。
- DSpark teacher projection cache 已按用户要求删除；没有数据长度分桶，也没有改变 batch/采样顺序。

目前还没有经核对的“所有优化之前”真实训练 baseline，因而没有全部优化的总加速百分比。


### 只在日志窗口计算指标（2026-10-14）

已实现，现有 log_interval 自动控制，无需新增配置。覆盖 DLite SFT/teacher/Stage-1/transition/Stage-2 的普通、hidden-cache、packet 路径，以及 DFlash、DFlash2、DSpark。非日志窗口返回训练 loss_terms，不计算准确率、接受长度、τ、teacher agreement、top-1 分布统计、confidence error 等诊断；CE/L1/KL/BCE、selector 目标、DSpark confidence 的 detached 接受概率目标及 loss denominator 仍按训练需求保留。

共享 runner 按 optimizer step 决定，日志窗口的每个梯度累积 microbatch 都收集指标。regular interval、scheduled final 和 partial accumulation tail 强制日志都覆盖。指标仍是该 optimizer 窗口的加权汇总；不改为单独最后一个 microbatch，也不改变日志频率。独立模型调用默认 collect_detailed_metrics=True，兼容原调用；训练 recipe 显式传入 runner 的决定。

DFlash 基础 gold probability/accuracy 和 DFlash2 selector 诊断现在也受开关控制；必须训练的 top-k/selector CE 保留，包括零 alpha warmup 的参数图。DSpark 的 confidence target 每步保留，只跳过其诊断消费者。checkpoint replay 跳过诊断，L1 top-1 maxima 移到 no_grad；编译 L1 的训练部分始终使用同一 reduction，避免日志开关切换训练内核或保存张量数量。没有引入 DSpark teacher cache。

测试：functional_checks.log 的相关检查 111 passed；cuda_loss_checks.log 中 logging on/off 的完整词表 confidence target、FP32/BF16 梯度、checkpoint 及 loss 回归 55 passed；补充 regular/tail/scheduled-final 日志窗口检查 6 passed；public_config_checks.log 通过。早期 checkpoint 保存张量数量不一致的失败记录保留，已修复并复测。新回归文件为 tests/test_metrics_logging_only.py，原 runner 实训测试扩展为日志开关两种情况。

正式训练对照：logging_metrics_warmup200_steps300/，相同代码/auto loss backend、相同编译缓存，按 all→logging→logging→all 四轮各 300 步；每轮前 200 步全部预热，使用共同的 98 个正常步骤。固定初始 FP32 master、input/anchors、batch、accumulation、loss、Muon/LR schedule。两 rank 四轮 weighted loss、grad norm 全部 300 步完全一致，最终所有 FP32 master hash 完全一致。log_interval=50 时，每种诊断每 rank 从 300 次降到 6 次（50/100/150/200/250/300），非日志步骤计数为 0。

合并窗口输入吞吐：每步指标 10731.87 → 仅日志指标 10932.24 tokens/s，本次观察 +1.87%。GPU 稳定窗口 PID/UUID 连续采样审计通过。

- all_a：step median 237.933 ms，mean 240.658 ms，input 10715.56 tokens/s，allocated 26.669 GiB、reserved 32.809 GiB。
- logging_b：step median 234.756 ms，mean 235.053 ms，input 10971.07 tokens/s，allocated 26.669 GiB、reserved 32.809 GiB。
- logging_b_repeat：step median 235.307 ms，mean 236.723 ms，input 10893.68 tokens/s，allocated 26.669 GiB、reserved 32.809 GiB。
- all_a_repeat：step median 238.139 ms，mean 239.927 ms，input 10748.23 tokens/s，allocated 26.669 GiB、reserved 32.809 GiB。

合并两轮关键 rank 时间线（含等待、通信，非纯 kernel 时长）：

- 加载数据：每步指标 1.758 ms → 仅日志指标 0.235 ms；新占比 0.10%。
- 前向准备：每步指标 14.254 ms → 仅日志指标 14.128 ms；新占比 6.00%。
- forward：每步指标 23.030 ms → 仅日志指标 22.972 ms；新占比 9.75%。
- loss及指标：每步指标 22.974 ms → 仅日志指标 21.364 ms；新占比 9.07%。
- backward：每步指标 69.886 ms → 仅日志指标 68.700 ms；新占比 29.17%。
- 参数更新：每步指标 108.033 ms → 仅日志指标 108.135 ms；新占比 45.91%。

吞吐含所有正常窗口的数据等待，不删除无新编译图的慢步骤。真实训练速度结论只覆盖这条双卡 offline DLite SFT；其他算法验证了功能、目标和梯度，不据此给出真实训练加速比。

额外复现检查表明，即使 torch 原生 loss，同一新代码在旧/新编译缓存下前三步也可出现不同 grad norm：torch_reference_old_cache3.json 与 torch_reference_shared_cache3.json。换回旧缓存复现旧运行 norm，新缓存的 torch reference 与本次 auto 的 norm 相同。因此本功能的数值验收使用相同代码、相同编译缓存的 on/off 对照；历史跨缓存轨迹不是开关的 baseline。

legacy_confidence_parity.json 记录旧 fused diagnostic reduction 与新固定训练 reduction 的额外比较：FP32 confidence target/leaf gradient 有最大 5.96e-8 舍入差，logit gradient 与 objective 无差；BF16 leaf gradients 在 probe 中完全一致。该记录不是声称旧/新所有 FP32 值逐位一致。当前 logging on/off 的 CUDA target/gradient 对照为严格逐元素一致。

代码和复现记录在 tests/metrics_feature_20261014/；真实训练每步数据、指标调用计数、phase event 计时、final master hash 和 GPU 审计在 tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/logging_metrics_warmup200_steps300/。没有 commit/push。
