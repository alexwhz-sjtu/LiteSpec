from pathlib import Path
root=Path('/diancpfs/user/wanghanzhen/LiteSpec')
p=root/'tests/training_optimization_handoff.md'
s=p.read_text()
marker='### 当前保留的优化与速度收益清单（2026-10-14）'
assert marker not in s
# The first eight sections are migration history. Make their old CE policy
# unambiguous to readers who enter the document at the shared-loss section.
needle='### CE / L1 / KL / BCE 公共 loss\n\n'
assert needle in s
s=s.replace(needle,needle+'**版本说明：本段及其表格记录 afbb9ed 迁移时的策略。2026-10-14 已撤回所有 dtype 的 compiled CE，当前 CE 保持原生，仅编译适用的 L1/KL/BCE 路径；最新策略与收益见本文末尾。**\n\n',1)
section='''
### 当前保留的优化与速度收益清单（2026-10-14）

本清单覆盖交接已有优化、本轮验证及本轮数值修复。当前 torch/auto ABBA 只隔离公共 loss backend；两组都启用 head 复用、参数 copy 等已有优化。因此它不是“所有优化相对于原始框架”的总对照，也不能累加各局部加速比。

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
- AdamW 没有启用 fused=True；没有全框架跳过非日志逐位置/τ 指标，也没有全面的 grad scale/clip/copy kernel 融合。
- DSpark teacher projection cache 已按用户要求删除；没有数据长度分桶，也没有改变 batch/采样顺序。

目前还没有经核对的“所有优化之前”真实训练 baseline，因而没有全部优化的总加速百分比。
'''
p.write_text(s+'\n'+section)
print('Updated',p)
