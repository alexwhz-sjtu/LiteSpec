"""文档占位；尚未集成实现。

Qwen3.5 GatedDeltaNet 占位，未来放入 backbone 的序列混合子层；暂不注册。
接收 draft_hidden_states；上下文注入方式仍须显式设计，输出维度恢复 hidden_size。
保留官方 in_proj_qkv / in_proj_z / in_proj_b / in_proj_a 的独立参数与 Q/K/V 顺序。
QKV 内部 causal depthwise conv、QK L2 norm、beta/g、gated RMSNorm 和 out_proj 是本组件数学；
不能替换为 operators/dflash2_conv，也不能当作 MHA/MLA 的 KV representation 或 access policy。
原生是因果递推，不能直接套用当前并行 block 的任意二维访问关系；多 anchor 隔离待设计。
实现前须核对 chunk/recurrent 前后向及 FLA 路径，当前无类、函数、导入或工厂。

参考说明：refrence/backbone/qwen35_integration.txt
官方源码：refrence/backbone/gated_deltanet.py
参考索引：refrence/index.html
"""
