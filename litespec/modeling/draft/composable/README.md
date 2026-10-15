# Composable draft v1

首版在原有 LiteSpec 上增加一个可组合的 draft、一个训练包装器和一个框架 adapter。
目标模型的 hidden_size、heads、head_dim、KV heads 等公共参数正常从 target config 继承。
模型配方只描述需要组合的组件；不重复手填 target 宽度。

## 组件和入口

- 可部署模型：`litespec/modeling/draft/composable/`。
- 训练包装器与 loss：`litespec/core/composable/`，各 loss 位于 `objectives/`。
- 框架 adapter：`litespec/training/recipes/composable.py`。
- 模型配方：`configs/composable_draft.json`；每一层对应 `backbone.layers` 的一项。
- 配置字段、选项与依赖：[configs/README.md](../../../../configs/README.md)。
- 训练入口：`python -m scripts.composable.train`。
- YAML 启动器：`python scripts/config/launch_composable.py --config scripts/config/composable.yaml`；可加 `--dry-run`。

Context 支持 `dflash`（concat → Linear → RMSNorm）和 `dflare`（每个 draft 层学习静态 target 层融合权重）。
DFlare 使用深度距离先验和独立 context/block K/V 投影。DFlare + MLA 使用独立 context latent 投影，属于组合扩展。

Backbone 支持 MHA（沿用 target KV heads，包含分组投影）和展开 KV 的 MLA。
Attention 的表示、访问方式，以及可选 DFlash2 卷积均可逐层配置。
MLA 的 rank、partial RoPE 和 value 维度属于该层的 MLA 内部参数，不修改公共 hidden_size。
`configs/` 的模型配方不包含 `backend` / `flex_backend`；计算方式通过运行 YAML 的
`arguments.attention_backend` 和 `arguments.flex_attention_backend`，或对应 CLI 参数选择。
支持 `eager`、`sdpa`、`flex_attention`；FlexAttention 的 kernel 选择为 `TRITON` / `FLASH`。
MHA/MLA 与 dense/sliding_window 四种组合均可使用 FlexAttention，访问语义不因 backend 改变。
FlexAttention 要求 dropout 为 0，QK/V head 维度至少为 16；FLASH 要求 PyTorch >= 2.11，当前验证环境使用 TRITON。
为兼容已有 checkpoint，解析后的配置仍可包含 backend 字段；旧配置里的值优先，
`--override-attention-backend` 可显式覆盖全部层。新模型配方省略这些字段，由运行参数统一注入。

Proposal 支持 `parallel`、`markov`、`rnn`、`sequential_rnn` 和 `selector`。
Markov、RNN 为附加 logits；sequential_rnn 使用 DLite 的顺序替换 head。
递归状态每个 block 重置。Selector 严格使用 unary top-K，按实际前驱逐步选择或采样。

本版不注册 target KV、KVShot、DSA、额外 confidence head、旧 CE+L1 组合。
原有算法及其数学实现保持原路径；新模型有独立注册、训练和导出分支。

## 每层配置

```json
{
  "block_size": 8,
  "composable_config": {
    "context": {"type": "dflare"},
    "backbone": {
      "type": "transformer",
      "layers": [
        {"attention": {"representation": "mha", "access": "dense"}},
        {
          "attention": {
            "representation": "mla",
            "access": "sliding_window",
            "window_size": 512,
            "kv_lora_rank": 512,
            "q_lora_rank": null,
            "qk_nope_head_dim": 64,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128
          },
          "local_operator": {"type": "dflash2_conv", "kernel_size": 2, "group_size": 16}
        }
      ]
    },
    "proposal": {"type": "rnn", "rank": 256},
    "objective": {"type": "nla", "loss_decay_gamma": 8.0}
  }
}
```

省略 target_layer_ids 时使用现有 DFlash 层选择策略；极浅 target 去重并限制在合法层内。
运行时 JSON 会保存解析后的具体 target 层和组件配置。已有 checkpoint 使用严格权重加载。

SWA 的 W 只计已确认历史：anchor=a 时历史范围为 [a-W,a)，另加完整当前 Q 个输入槽位。
块内双向可见；同一 block 的所有 query 使用相同历史范围。
各 block 互相隔离；提供 document IDs 时还会隔离不同文档。
MLA 的展开 KV 仍按其既有 query-head 组织。推理缓存每个样本物理保留至多 W 个已确认 KV，保留绝对位置。临时 draft block 不写入已确认缓存。

## Loss 的原生尺度

- `hard_ce`：对已存储的 target token 做 CE；有效位置固定权重之和作分母。
- `soft_ce`：teacher 分布的交叉熵；`forward_kl` 使用 KL(p||q)，保留其标量与 Soft CE 的差异。
- `tv`：0.5 × sum |p-q|，与旧算法未除以 2 的 L1 区分。
- `nla` / `lk_alpha`：-log(sum min(p,q))，overlap 下限 1e-5；
  每个原始样本先除以有效预测数 + 1e-5，再按样本平均。
  固定位置 decay 仅进入分子，不能放入该计数分母。这里的 α 是 acceptance，不新增一个 α 超参数。
- `dpace`：保留原始 dtype 的 unsmoothed CE、平滑概率、无梯度 prefix/suffix credit，
  最后除以原始 batch size。配置 `alpha` 默认为 0.5；不叠加固定位置衰减。
- `bv`：vocabulary-integrated prefix score + annealed block-log。
  沿 target 采样路径更新 ratio/acceptance，保留前缀梯度；只统计完整连续有效预测块。
  B 不包含 anchor。前三 epoch 的 beta 从 0 线性升到 1。
  **纯 parallel proposal 的退火 score 下限为 1e-6；markov/rnn/sequential_rnn 不加该下限。**
  正式 block-log 的 log summand 下限 -80、length 下限 1e-6 保持原配方。
  日志将数学目标与训练 gradient surrogate 分开。BV + selector 明确报错。

NLA 对齐冻结的 Speculators 公共实现，不声称它是 LK 论文作者代码。
D-PACE 对齐冻结的作者仓库片段；BV 作者源码尚未定位，因此标记为论文复现。
冻结片段、来源 revision 和许可证均保留于 `refrence/`；部署代码不导入该目录。
其他组件直接复用现有 LiteSpec 对应数学原语，新增包装只管理接口、访问和组合。
通用 anchor sampling/gather 复用 LiteSpec；不复制上游训练脚本中的数据准备和 anchor 小技巧。

Selector 的主目标使用 base_logits；辅助目标为覆盖候选的 CE。
未命中候选的位置无辅助 CE 梯度，仍参与主目标。
辅助 CE 分母为所有有效位置的原始固定权重和，不改为命中数。
固定位置 decay 默认沿用 objective 的 loss_decay_gamma；proposal 显式配置该字段时覆盖，显式 null 可关闭辅助位置衰减。
主目标与辅助目标有不同分母时，先准备整个 optimizer window 的 anchors/masks，
跨数据并行组统计原生分母，再等价转换交给现有 runner。
原生分母由 Objective.native_denominator 提供；selector 辅助 CE 也独立放在 objectives/selector.py，adapter 不包含 loss 公式。
不平均局部 mean；分块不切断 D-PACE/BV 的预测链，不重复累计原始 N。

## 训练与数据

默认 YAML 使用现有 Qwen3-4B checkpoint 和已有生成数据，仅默认 Hard CE。
online、regen_full offline、FSDP、SGLang target TP、现有 shard_draft_by_tp 样本分工、
target/draft 分离部署均通过相同 adapter 接入。
没有新增切分 draft 参数的张量并行，也没有放宽现有离线/分离部署/TP 组合限制。

使用 BV 前核对轨迹采样与 teacher 分布一致。
首版按完整词表、temperature=1 的 target 轨迹验证，并要求显式
`--bv-sampling-provenance target_full_vocab_t1`，把这项数据声明写进 checkpoint。
不能把已有 top-p 截断数据直接声明为这种论文复现设置。

四个模块的配方保存在 JSON；attention backend、数据、batch、优化器、拓扑等放在 YAML/CLI。
Selector alpha、warmup 和 ramp 使用现有运行参数。BV 的 epoch progress 来自 runner 游标。
objective_chunk_blocks 只沿 anchor 维度切块；0 表示不切。
混合目标预取缓冲通过 window_buffer_mb 控制，超出部分转存到 cache_dir/composable_windows；
当前正在执行的一个 batch 另占内存。接收槽在推进前复制，防止通信 buffer 复用覆盖。
恢复先恢复游标/RNG，再准备本窗口；checkpoint 只在完整 optimizer 边界保存。

## 模型调用与缓存

模型与训练均经过 Context → Backbone → Proposal。
`mode="features"` 返回 backbone 的 B 个预测状态，同时保留完整 backbone_hidden_states，供训练包装器在 FSDP 内对真实 Q 槽位分块调用同一个 Proposal；
`teacher_forcing` 使用指定 previous_token_ids；`sample` 根据真实已生成前驱递推。

ModelInputs：
target_hidden_states=[N,S,T*D] 或 [N,S,T,D]；
draft_embeddings=[N,A*Q,D]，每块首槽为 clean anchor，其余为 MASK；
anchor_positions、block_keep_mask=[N,A]；
previous_token_ids=[N,A,B]。
训练 adapter 沿用现有单条对话/样本数据路径，不新增 packed 数据预处理。

所有位置均为绝对位置。自定义分段输入需要同时提供 document_ids 和 anchor_document_ids。
Cached inference 每个样本一次处理一个当前 block；使用 DraftState，
仅传入新增且已确认的 context，并提供实际 context_position_ids。
重复传入已缓存位置会报错；rollback(next_position) 可使用标量或每样本阈值。
SharedOps 绑定冻结 target embedding/head，不复制这些权重到可部署 draft。

DraftOutput 区分 draft_hidden_states、base_logits、draft_logits、candidate_logits 和 draft_probabilities。
Selector 的 teacher-forcing 输出提供 base/candidate logits，训练包装器显式选择 base 主目标；
sample 模式返回沿真实采样前缀的完整稀疏分布及对应 dense logit 表示。
顺序替换 head 不额外投影一份无用 unary logits。

模型只负责提议，采样验证与 target 状态管理仍由推理侧负责。
BV 的效果评估需要匹配 block verification；不能把训练 loss 数值直接解释为逐 token 验证吞吐。

## 导出与验证

checkpoint 包装器成员保持 draft_model；可训练参数全在前三模块，Objective 无参数。
FSDP 不将原生计数/窗口分母转换成 BF16。
专用导出器生成 inference-only flat dependency bundle，保留开发源码的分包组织。
AutoConfig / AutoModel 冷启动无需导入 LiteSpec、训练模块或 refrence；
调用时仍需绑定 SharedOps。导出源文件保留原文件的许可证说明。

测试和所有小模型、合成数据、日志、导出产物均位于
`tests/composable_implementation_20261015/`。
查看 `validation.json` 获取最终执行结果；`run_runtime_check.py` 可复现各运行链路。
这些是短训练、公式/梯度与恢复测试，不是大模型吞吐或接受率的性能结论。

本次验证解释器：/data/wanghanzhen/venvs/FlashMTP_v2.3/bin/python。测试工具仅安装在 tests 下的 test_dependencies，未修改共享训练环境。
