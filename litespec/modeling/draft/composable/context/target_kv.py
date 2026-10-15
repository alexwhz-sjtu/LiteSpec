"""文档占位；尚未集成实现。

KVShot KV-only parallel adapter。逐 token 多层 target K/V 拼接，分别投影为每个 draft 层的 expanded context KV。原文定义与待移植内容见 refrence/context/kvshot_kv_only.paper.txt。
层选择和dflash选择策略一样
实施计划：output/draft-research/LiteSpec-draft-modules-research.html
参考索引：refrence/index.html
"""
