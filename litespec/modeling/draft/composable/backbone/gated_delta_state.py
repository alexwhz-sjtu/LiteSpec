"""文档占位；尚未集成实现。

GatedDeltaNet 状态契约文档占位；未来作为 DraftState.backbone_state 子状态。
每层独立保存 conv_states [N, 2*Hk*Dk+Hv*Dv, Kconv] 和 recurrent_states [N,Hv,Dk,Dv]；
多 anchor 的 N 展开须明确且状态隔离，不能把不同 anchor 当作同一序列直接串联。
混合 full-attention 层另存普通 KV；state orientation、dtype、已确认位置需显式记录。
投机拒绝时须同时恢复卷积和递推状态；普通 KV crop 不适用，待选快照/重放等策略。
固定 v5.3.0 参考在 cached seq_len>1 时会走 initial_state=None 的 chunk 分支；
不能据此宣称已经支持从已有 state 续算整个 draft block。本文件不实现缓存或回滚。

参考说明：refrence/backbone/qwen35_integration.txt
官方源码：refrence/backbone/gated_deltanet.py
参考索引：refrence/index.html
"""
