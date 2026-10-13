# LiteSpec 训练优化交接文档

记录日期：2026-10-13。源码位置：`/data/wanghanzhen/projects/SpecDecoding/LiteSpec`。

本文对应当前工作树，供迁移机器后继续实验、检查数值和修改代码使用。实现及小规模正确性检查已完成，**最终实现的可靠性能测量尚未完成**：旧机器存在其他任务占用，已有 GPU 计时受干扰，而且旧测量早于最后一次 L1 精度修正。

## 1. 用户已经确定的范围

- 优化范围覆盖 DLite、DFlash、DFlash2、DSpark 的相关公共训练路径。
- 不做长度分桶，不改变 batch size、采样顺序或数据组织。
- 处理 DLite transition 重复 head、DFlash 系列日志重复投影、FP32→BF16 显式临时张量，以及公共 loss 融合。
- **不要给 DSpark 增加 teacher 投影缓存。** 曾实现的 `_TeacherReplayCache` 和 `LITESPEC_TEACHER_CACHE_MIB` 已按用户要求删除，对应缓存测试也已删除。后续不要把它们作为现有功能恢复。
- 不以共享 GPU 上的计时得出加速结论；先保证数值，再在空闲机器上验证收益。

此前还讨论过 fused AdamW、grad norm、full-attention mask 和 ptxas 探测。本文只将核对过的修改列为完成项：当前 `BF16Optimizer` 的 `torch.optim.AdamW(...)` **没有显式传入 `fused=True`**；没有据此声称已启用 fused AdamW。本轮也没有把全框架非日志 step 跳过所有逐位置/τ 指标作为已完成项。

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
