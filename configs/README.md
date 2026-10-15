# Composable draft 模型配置说明

本文对应 `composable_v1` 的实际实现，说明本目录模型配方的字段、默认值、所有已支持选项，以及字段之间的依赖。配置为标准 JSON，不支持注释、模板、`repeat` 或其他 DSL 展开语法。

## 1. 模型配方与运行配置

本目录的 JSON 描述四个模块：`context → backbone → proposal` 构成可部署模型，`objective` 描述训练目标。Attention 是 backbone 内部组件。

- 模型配方：本目录的 `composable_*.json`。
- 运行配置：[scripts/config/composable.yaml](../scripts/config/composable.yaml)，包含 backend、数据、优化器、batch、拓扑和训练调度。
- 训练入口：`python -m scripts.composable.train --draft-config configs/composable_draft.json ...`。
- 配置加载：[training/recipes/composable.py](../litespec/training/recipes/composable.py)。

本目录 JSON 不填写 `backend` / `flex_backend`。计算方式不属于模型架构，详见第 9 节。

当前配方示例：

- [composable_draft.json](composable_draft.json)：DFlash + 5 层 MHA/dense + parallel + Hard CE。
- [composable_heterogeneous.json](composable_heterogeneous.json)：DFlare + 逐层 MHA/MLA、dense/SWA、局部卷积 + RNN + NLA。具体参数以文件为准。
- [composable_selector_dpace.json](composable_selector_dpace.json)：DFlare + MHA/dense + selector + D-PACE。
- [composable_bv.json](composable_bv.json)：DFlash + MHA/dense + parallel + BV。

## 2. 顶层字段与继承规则

### `architectures`

字符串数组。本目录统一写 `["ComposableDraftModel"]`，表示模型类名，不是组件选择器。训练 adapter 会构造并保存该模型类的配置；从头训练时不靠修改本字段切换算法。可省略。

### `block_size`

整数，必须至少为 2。一个 draft block 的总输入槽位数 **Q = 1 个 anchor + B 个预测槽位**，因此 `B = block_size - 1`。

- 本目录示例为 8，即预测 7 个 token。
- 从头训练：JSON 中的值优先；省略时使用运行参数 `--block-size`，默认 8。
- 直接创建 `ComposableDraftConfig` 时默认也为 8。
- 所有层、proposal 和 objective 共享同一个 block size，不能逐层指定。
- 卷积的 `kernel_size` 不能超过 Q；sequential RNN 的最大递推长度由 B 自动确定。

### `composable_config`

对象，包含 `context`、`backbone`、`proposal`、`objective` 四个对象。四者在手写配方中均应提供。适配器也接受直接以这四个对象为根的文件，但推荐保留此包装，便于区分顶层 `block_size`。

`composable_config.mask_token_id` 是可选的运行解析/持久化字段：表示现有 target embedding 的一个有效行号，要求 `0 <= id < vocab_size`。通常不手填，由运行系统决定并写回 checkpoint：显式 `--mask-token-id` > 配方/已有 checkpoint 的值 > tokenizer 的 mask token > tokenizer 未使用的模型词表行。它不是额外扩展词表的开关。

### 从 target 继承的公共配置

从头训练时，适配器先读取 target 的 text config，再确定 draft 深度与四模块配方。公共参数无需在本目录 JSON 中重复指定：

- `hidden_size`：所有层输入、输出与 residual 的公共宽度。
- `num_attention_heads`：attention query heads。
- `num_key_value_heads`：MHA/GQA 的 KV heads。
- `head_dim`：MHA 的每个 head 宽度；保留 target 实际配置的显式值，不能自行以 `hidden_size / num_attention_heads` 替代。
- `intermediate_size`、`hidden_act`：当前 FFN 的中间宽度与激活。
- `rms_norm_eps`：RMSNorm 的 epsilon。
- `attention_bias`、`attention_dropout`：沿用的 attention 属性；FlexAttention 要求 dropout 为 0。
- `initializer_range`：初始化尺度，也用于 selector codebook。
- `vocab_size`、位置编码/最大位置相关配置及其他标准 Qwen3 配置：正常继承。

派生字段：

- draft 的 `num_hidden_layers = len(backbone.layers)`，训练 adapter 自动设置。
- `num_target_layers` 保存 target 深度，供 target 层选择与检查使用。
- `model_type = "litespec_composable"`、`schema_version = "composable_v1"` 由配置类设置。

这些字段会出现在 checkpoint 的完整 `config.json` 中，但不是本目录配方的手工覆盖入口。**当前 adapter 不会把手写配方顶层的任意 HF 字段合并到 target config**；例如，在配方顶层填写 `hidden_size` 不会覆盖继承的宽度。`--num-draft-layers` 也不覆盖显式 `layers` 列表。

MHA 的 query heads 必须能按 KV heads 分组；MLA 的展开 KV 按 query heads 组织，不直接使用 target 的 MHA KV 布局。公共 hidden 宽度一致，不代表不同 attention 的内部 Q/K/V 维度必须一致。

## 3. Context：target 信息注入

路径：`composable_config.context`。

### `type`：必填

所有选项：

- `"dflash"`：拼接选中的 target hidden，经过无 bias Linear 和 RMSNorm；得到的 context hidden 供所有 draft 层使用。文件名 `hidden_mlp.py` 不表示此处额外包含非线性 MLP。
- `"dflare"`：每个 draft 层分别学习 selected target layers 的 softmax 融合权重，再做 RMSNorm。权重为静态可训练参数，不是每个 token 动态生成的 gate；初始化使用层深度距离先验。Attention 使用独立的 context/block KV 投影；MLA 对应独立 context latent 投影。

两者都可以搭配 MHA/MLA、dense/sliding_window 及所有已支持 proposal；本版不支持 `target_kv`、KVShot 或 target MHA KV 到 draft MLA KV 的迁移。

### `target_layer_ids`：可选

target transformer 层的零起始整数索引数组，例如 `[1, 9, 17, 25, 33]`。

- 必须非空、升序、无重复，且全部位于 `[0, num_target_layers)`。
- 表示每层输出 hidden；取 Hugging Face `hidden_states` 时使用 `layer_id + 1`，不把 embedding 输出当作第 0 个 transformer 层。
- 数量不要求等于 draft 层数；DFlash 的拼接投影输入宽度会依此变化，DFlare 的融合权重列数也会变化。
- 省略或为 `null`：模型创建时沿用既有 DFlash 层选择。单个 draft 层取 target 中部；多层通常在 target 的第 1 层到倒数第 3 层之间均匀选取，再排序去重。target 少于 4 层时选择全部层。
- 非空 CLI `--target-layer-ids` 优先于配方中的值，仅适用于从头构造配置的路径。

改变此字段会改变融合参数形状或语义；不能当作无关的运行参数随意套用旧权重。

## 4. Backbone 与逐层配置

路径：`composable_config.backbone`。

### `type`

当前唯一选项为 `"transformer"`；省略时按 transformer 处理。

### `layers`：必填

非空数组，数组顺序就是执行顺序，每个元素描述一层。可以任意交错已支持的 attention 表示、访问方式和卷积。

每层包含：

- `attention`：必填对象，详见第 5 节。
- `local_operator`：可选对象，详见第 6 节。

各层使用相同的公共 hidden size、heads、FFN、norm 和 block size。目前不能在层对象内覆盖这些公共属性，也不能通过配置重新排列 norm、residual 或 attention/FFN 顺序。

直接构造完整 `ComposableDraftConfig` 时，`len(layers)` 必须等于 `num_hidden_layers`；使用训练 adapter 时会自动从数组长度推导后者。

## 5. Attention 字段

路径：`composable_config.backbone.layers[i].attention`。以下参数均属于第 i 层。

### `representation`：必填

所有选项：

- `"mha"`：标准 Q/K/V 投影，保留 target 的 query heads、KV heads 和显式 `head_dim`；KV heads 少于 query heads 时实际为 GQA 分组布局。
- `"mla"`：共享低秩 KV latent 与 partial RoPE；可以选择低秩 Q。首版仍展开 K/V 后计算并缓存，不包含压缩 latent cache 优化。

### `access`

所有选项：`"dense"`、`"sliding_window"`；默认 `"dense"`。

- dense：访问该 anchor 之前的全部有效历史 context，以及完整当前 block。
- sliding_window：访问该 anchor 之前最近 W 个位置范围内的有效历史 context，以及完整当前 block。
- 两者的当前 block 都双向可见；不同 draft block 相互隔离。有 document 信息时还会隔离不同文档。
- 这不是 causal-block attention，也不会随 block 内 query 位置移动历史窗口。

### `window_size`

仅 `access = "sliding_window"` 时必填，为正整数 W。历史位置范围为 `[anchor - W, anchor)`；W 不包含当前 block 的 Q 个槽位。Persistent inference cache 每个样本物理保留至多 W 个历史 KV，并保留绝对位置。

`access = "dense"` 时本字段不生效，应省略。

### MLA 专用维度字段

仅 `representation = "mla"` 时生效；MHA 层应省略。

- `kv_lora_rank`：必填正整数，压缩 KV latent 的宽度。
- `qk_nope_head_dim`：必填正整数，每个 query/key head 不做 RoPE 的维度。
- `qk_rope_head_dim`：必填正偶数，每个 query/key head 做 RoPE 的维度。
- `v_head_dim`：必填正整数，每个展开 value head 的宽度。
- `q_lora_rank`：可选，`null` 表示直接 Q 投影，正整数表示 `Q down-projection → norm → up-projection` 的低秩 Q 路径。省略时沿用继承配置的同名属性；target 无此属性时相当于 `null`。要明确关闭低秩 Q，可显式写 `null`。
- `rope_interleave`：可选布尔值。`true` 使用交错旋转，`false` 使用 NeoX 风格的分半旋转；省略时沿用继承配置同名字段，无此字段时默认 `true`。

依赖关系：

```text
MLA QK head dimension = qk_nope_head_dim + qk_rope_head_dim
MLA V head dimension  = v_head_dim
MLA output projection: query_heads * v_head_dim → hidden_size
```

以上内部维度不要求等于 target 的 MHA `head_dim`，最终输出宽度仍必须为公共 `hidden_size`。

## 6. 每层局部算子

路径：`composable_config.backbone.layers[i].local_operator`。

### `type`

所有选项：`"none"`、`"dflash2_conv"`。整个对象省略时等价于 `{"type": "none"}`；如果对象存在，应提供 `type`。

- none：不添加局部算子。
- dflash2_conv：复用 DFlash2 grouped dynamic depthwise convolution；在该层 attention 与 FFN 两个子层分别建立独立卷积模块，并保留原生 prepare/finish/residual 顺序。卷积在各 block 内执行，不混合不同 anchor block。

### `kernel_size`

仅 dflash2_conv 生效。默认 2，整数，要求 `1 <= kernel_size <= block_size`。

### `group_size`

仅 dflash2_conv 生效。默认 16，表示每个通道组包含的 hidden channels 数量，**不是组数**。必须为正整数且整除 `hidden_size`；组数自动计算为 `hidden_size / group_size`。

Conv 可用于 MHA/MLA、dense/sliding_window；同一层的 attention 与 FFN 卷积共享配方参数，但不共享权重。

## 7. Proposal：候选生成与依赖补偿

路径：`composable_config.proposal`。

### `type`：必填

所有选项：

- `"parallel"`：直接使用 backbone 的 B 个预测 hidden，通过冻结 target LM head 并行生成 logits；无额外依赖 head。
- `"markov"`：DSpark Markov head，根据前驱 token 产生附加 logits，与基础 logits 相加。
- `"rnn"`：DSpark RNN Markov head，结合 draft hidden 和前驱信息递推，产生附加 logits；基础 logits 仍保留。
- `"sequential_rnn"`：DLite sequential RNN head，以顺序 head 替代基础词表预测，不再额外计算并相加一份基础 LM-head logits。
- `"selector"`：DFlash2 selector。先取每个位置基础 logits 的 unary top-K，再根据实际前驱逐位置打分和选择；不是全局 Viterbi 搜索。

递推状态每个 block 重置。训练 teacher forcing 的前驱来自真实轨迹，生成时使用实际生成的前驱。

### `rank`

可选正整数，默认 256。

- markov/rnn：Markov correction 的低秩宽度。
- sequential_rnn：DLite sequential head 的 latent/state 宽度。
- selector：候选转移 scorer 的 state rank。
- parallel 不使用，应省略。

### `top_k`

仅 selector 生效，默认 16，整数且 `1 <= top_k <= vocab_size`。表示每个位置基础 logits 选出的候选数。改变它不扩充或缩小模型词表。

### `stop_gradient`

仅 selector 生效，默认 `false`，布尔值。设为 `true` 时，selector 辅助 CE 不沿 unary candidate logits 或输入 hidden 回传到基础模型；selector 自身仍可训练，主目标仍训练基础模型。此开关不是冻结整个 proposal。

### `loss_decay_gamma`

仅 selector 辅助 CE 生效，取值语义同 objective 的同名字段。

- 不填写：继承 `objective.loss_decay_gamma`。
- 显式正数：覆盖辅助 CE 的位置衰减。
- 显式 `null` 或非正数：关闭辅助 CE 的位置衰减，即使 objective 使用了正数。

主目标与 selector 辅助 CE 独立计算、独立归一化。selector 下主目标使用 base logits；辅助 CE 只对真实 token 被 top-K 覆盖的位置产生梯度，分母仍统计所有有效位置的原始固定权重，不改为命中数。

辅助 CE 的整体系数与启用调度放在运行配置：`selector_loss_alpha`、`selector_warmup_ratio`、`selector_ramp_ratio`，不属于本目录模型 JSON。

## 8. Objective：训练目标

路径：`composable_config.objective`。Objective 无可训练参数；selector 等可训练头保留在模型中。推理模型源码不依赖 Objective 实现。

### `type`：必填

全部合法字符串：`"hard_ce"`、`"soft_ce"`、`"forward_kl"`、`"tv"`、`"nla"`、`"lk_alpha"`、`"dpace"`、`"bv"`。`lk_alpha` 与 `nla` 是同一个实现的两个名称，因此共六类方法。

- hard_ce：真实轨迹 token 的 CE；不需要完整 teacher 概率分布。
- soft_ce：完整 teacher 分布 p 对 draft q 的交叉熵 `-sum(p * log(q))`。
- forward_kl：`KL(p || q)`。和 Soft CE 对 draft 的梯度一致，但记录的标量相差 teacher entropy。
- tv：`0.5 * sum(abs(p - q))`，不是未乘 0.5 的 L1。
- nla / lk_alpha：`-log(sum(min(p, q)))`，overlap 下限固定 `1e-5`。对齐冻结的 Speculators 公共实现，不声称为 LK 作者源码；名称中的 acceptance α 不是待填写的超参数。
- dpace：CE 乘以由本模型 label 概率计算的无梯度 prefix/suffix credit；保留原生 CE dtype 和归一化，不需要完整 teacher 概率分布。
- bv：vocabulary-integrated prefix score 与退火 block-log。当前为论文复现，并非已定位作者源码的实现。

soft_ce、forward_kl、tv、nla/lk_alpha、bv 均需要冻结 teacher logits，数据/target 路径必须提供相应 target last hidden 与 LM head。Hard CE、D-PACE 仍需要正确的轨迹 token labels。

### `loss_decay_gamma`

可选数值或 `null`，默认 `null`。为正数 γ 时，对 block 内第 j 个预测位置乘 `exp(-j / γ)`，j 从 0 开始，不包含 anchor。γ 越小越偏重前部；省略、`null`、0 或负数均不衰减。

- hard_ce、soft_ce、forward_kl、tv：使用该权重，最终除以有效位置权重和。
- nla/lk_alpha：权重只进入分子；先按每个原始样本的有效预测数加 `1e-5` 归一化，再按原始 batch size 平均。不能把位置权重和放进其分母。
- dpace：主目标不读取该字段，使用原生 credit；若搭配 selector，该字段仍可作为辅助 CE 的默认衰减。
- bv：主目标不读取该字段；使用自身完整 block 归一化。

本版不存在可配置的 `reduction`、soft-target temperature、NLA epsilon 或 BV 退火超参数字段。

### `alpha`

仅 dpace 生效，默认 0.5，数值且 `0 <= alpha <= 1`。原生平滑概率为 `(1 - alpha) * p(label) + alpha`，用于构造无梯度的 credit，最后 loss 除以原始 batch size。它不是 selector loss 系数，也不是 NLA 中的 acceptance α。

### BV 的固定规则与依赖

- `proposal.type = "selector"` 时禁止使用 BV；其余四种 proposal 已支持。
- 仅完整、连续、全部 B 个预测位置都有效的 block 参与目标，分母为这些 block 的数量。
- 前 3 个 epoch，`beta = clamp(epoch_progress / 3, 0, 1)`；epoch progress 来自训练 runner。
- 退火阶段 beta < 1：parallel proposal 的 score 下限为 `1e-6`；markov/rnn/sequential_rnn 的 log-score 不使用该下限。由 proposal 类型自动决定，不手填 floor 字段。
- 正式 block-log 阶段：log summand 下限 -80、length 下限 `1e-6` 固定保留。
- 运行参数必须声明 `bv_sampling_provenance: target_full_vocab_t1`。声明前需确认数据轨迹确实来自完整 target 词表、temperature=1 的采样，且 teacher 对应同一分布；此参数不会自动转换已有截断采样数据。

## 9. 计算方式与其他运行参数

下列字段填写在运行 YAML 的 `arguments` 中，或使用对应 CLI 参数；不要放入本目录模型 JSON。

```yaml
arguments:
  draft_config: configs/composable_heterogeneous.json
  attention_backend: flex_attention
  flex_attention_backend: TRITON
```

### `attention_backend`

选项为 `eager`、`sdpa`、`flex_attention`。Composable 训练入口默认 sdpa。

- eager：显式 scores、softmax、V，便于对照。
- sdpa：PyTorch scaled_dot_product_attention。
- flex_attention：PyTorch FlexAttention，kernel 由下一个运行字段选择。

MHA/MLA × dense/sliding_window 四种组合都可以使用上述三个接口；具体 kernel 仍受输入维度、软件版本与硬件限制。Backend 不改变 block visibility 或 SWA 的定义。

### `flex_attention_backend`

选项为 `TRITON`、`FLASH`，默认 TRITON；仅 flex_attention 生效。

- TRITON：当前 PyTorch 2.9.1 环境下已完成四种 attention 组合的 loss/梯度对照，包括 DFlash 与 DFlare 两种 context。
- FLASH：现有 runtime 明确要求 PyTorch >= 2.11；当前环境不支持该分支。
- FlexAttention 还要求 `attention_dropout = 0`，QK 和 V 的 head dimension 都至少为 16。MHA 使用继承的 head_dim；MLA 分别检查 `qk_nope_head_dim + qk_rope_head_dim` 与 `v_head_dim`。

没有独立的 `flash_attention_2` / `flash_attention_3` 选项。Flex 的 FLASH 是现有 Flex kernel 选择，不是这两个接口。

### 与配方相关的其他运行字段

- `override_attention_backend`：可选 eager/sdpa/flex_attention，用于覆盖旧配置中已存在的逐层 backend。新配方没有 backend 字段，正常使用 attention_backend 即可。
- `selector_loss_alpha`：默认 1.0，非负，辅助 CE 总系数；0 会冻结 selector 参数。
- `selector_warmup_ratio`：默认 0，范围 `[0,1]`，按总 optimizer steps 的比例，先保持辅助系数为 0。
- `selector_ramp_ratio`：默认 0，范围 `[0,1]`，warmup 后线性增大到目标系数所用步数比例。
- `objective_chunk_blocks`：默认 32，非负整数，沿 anchor 维度分块，0 关闭分块；不切断单个 block 内的递推/credit 链。
- `window_buffer_mb`：默认 256，正数，混合目标 optimizer-window 预取缓冲预算，超出转存到 cache；不改变模型数学定义。
- `num_anchors`：Composable 默认 32，训练 anchor 数；不会改变模型层数或 block_size。
- `target_layer_ids`：逗号分隔的 target 层索引字符串，从头训练时覆盖 context.target_layer_ids。
- `mask_token_id`：可选有效词表行号，优先级见第 2 节。
- `bv_sampling_provenance`：当前唯一合法值 target_full_vocab_t1，BV 必需，其他 objective 不要求。

从头训练的 backend 解析顺序：已有旧层 backend 值 > attention_backend 默认注入；显式 override_attention_backend 再覆盖全部层。Flex kernel 同样优先保留旧层 flex_backend，否则由运行参数注入。

为兼容已有 checkpoint，解析后的完整 config.json 仍可保存 backend/flex_backend；它们是执行设置的记录，不是本目录配方要求用户填写的架构字段。

## 10. 依赖与冲突速查

1. `block_size >= 2`；所有层共用；预测长度为 block_size - 1。
2. `backbone.layers` 决定 draft 深度，不由 CLI num_draft_layers 决定；层输入输出公共宽度保持一致。
3. 每层必须提供 attention.representation；MLA 必须提供四个内部维度参数，RoPE 维度为正偶数。
4. access=sliding_window 必须有正 window_size；dense 不使用 window_size。
5. local_operator=dflash2_conv 时 kernel_size 受 block_size 限制，group_size 受 hidden_size 限制。
6. proposal=parallel 不使用 rank；其余 proposal 使用正 rank。仅 selector 使用 top_k、stop_gradient 和 proposal.loss_decay_gamma。
7. objective=bv 与 selector 冲突；其退火 floor 自动依赖 proposal 类型。
8. objective=dpace 才使用 objective.alpha；nla/lk_alpha 不接收独立 alpha 超参数。
9. proposal.loss_decay_gamma 缺省继承 objective；显式 null 关闭辅助衰减。D-PACE/BV 主目标不应用固定位置 decay。
10. Soft CE/KL/TV/NLA/BV 依赖 teacher 分布；BV 还依赖正确的采样轨迹来源。
11. FlexAttention 不要求特定 representation/access，但要求可运行的 head dimensions、零 dropout，以及对应 kernel 的软件/硬件条件。
12. Context 的 target_layer_ids 必须合法且匹配输入 hidden 的层数和顺序；输入特征宽度随所选层数变化。

当前不支持 target KV 注入、DSA、任意 per-layer hidden/head 数、可配置 FFN/norm/residual 拓扑、MLA 压缩缓存，也没有任意多个 proposal/objective 的自由列表组合。

## 11. 一个完整的最小示例

下面是可从头训练的两层异构配方。公共宽度从 target 继承；backend 在运行 YAML 中选择。

```json
{
  "architectures": ["ComposableDraftModel"],
  "block_size": 8,
  "composable_config": {
    "context": {"type": "dflare"},
    "backbone": {
      "type": "transformer",
      "layers": [
        {
          "attention": {"representation": "mha", "access": "dense"}
        },
        {
          "attention": {
            "representation": "mla",
            "access": "sliding_window",
            "window_size": 512,
            "kv_lora_rank": 512,
            "q_lora_rank": null,
            "qk_nope_head_dim": 64,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "rope_interleave": true
          },
          "local_operator": {
            "type": "dflash2_conv",
            "kernel_size": 2,
            "group_size": 16
          }
        }
      ]
    },
    "proposal": {"type": "parallel"},
    "objective": {"type": "hard_ce", "loss_decay_gamma": null}
  }
}
```

此示例要求 target.hidden_size 能被 16 整除；若不能，修改 group_size 或删除 local_operator。无需要求 target.head_dim 等于 MLA 内部 QK/V 维度。

## 12. 加载、恢复与当前校验边界

从头训练使用 target config + 本目录配方 + 运行参数，解析后保存完整配置。`--init-from` 或 `--resume-from` 路径则直接读取 checkpoint 自带配置与权重，**不会重新合并 --draft-config，也不会通过此处 build_config 重新注入 backend**；不要以为修改 JSON 就能改变正在恢复的模型结构。权重严格检查 missing/unexpected/mismatched keys，恢复训练还会核对配方与训练元数据。

约束分布在配置验证器、组件构造函数和运行初始化中，尚未统一为严格 JSON Schema。当前并非所有未知字段都会报错：例如 MHA 上填写 MLA 参数、dense 上填写 window_size、parallel 上填写 rank、非 selector 上填写 top_k，可能被忽略；部分数值类型也会被 int/float 转换。应仅填写本文说明适用于该组件的字段，且使用规范 JSON 类型。本文的正数/布尔值等要求同时描述合法使用语义，不代表所有情况都已由配置类提前拒绝。

实现入口：

- [配置与组合检查](../litespec/modeling/draft/composable/configuration.py)
- [模型与 target 层选择接入](../litespec/modeling/draft/composable/modeling.py)
- [Context 实现](../litespec/modeling/draft/composable/context/)
- [Layer 与 attention 参数传递](../litespec/modeling/draft/composable/backbone/layer.py)
- [Attention 与访问策略](../litespec/modeling/draft/composable/backbone/attention/)
- [卷积原语](../litespec/modeling/draft/dflash2.py)
- [Proposal 实现](../litespec/modeling/draft/composable/proposal/)
- [各类 Objective 与 selector 辅助目标](../litespec/core/composable/objectives/)
- [运行配置解析和训练适配](../litespec/training/recipes/composable.py)
- [FlexAttention runtime](../litespec/modeling/draft/flex_attention.py)
- [模型训练/缓存整体说明](../litespec/modeling/draft/composable/README.md)
