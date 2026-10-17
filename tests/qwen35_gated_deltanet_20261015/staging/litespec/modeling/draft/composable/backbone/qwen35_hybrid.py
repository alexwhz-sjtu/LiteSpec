"""文档占位；尚未集成实现。

Qwen3.5 hybrid backbone 文档占位；保持四个顶层模块，不注册新组件。
未来按 layer_types 编排 linear_attention / full_attention，3:1 只是官方默认模式。
full_attention 使用官方 gated attention；不能默认复用现有 MHA 就等价。
保留 pre-norm -> mixer -> residual -> post-norm -> FFN -> residual。
纯 GatedDeltaNet 子层和整套 Qwen3.5 hybrid 是不同可选范围；层数和 hidden_size 由 draft 配方决定。
context/proposal/objective 仍由原有边界负责；兼容性未证明前不能宣称任意 Lego 组合可用。

参考说明：refrence/backbone/qwen35_integration.txt
官方源码：refrence/backbone/gated_deltanet.py
参考索引：refrence/index.html
"""
