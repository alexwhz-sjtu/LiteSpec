from pathlib import Path
p=Path('/diancpfs/user/wanghanzhen/LiteSpec/tests/training_optimization_handoff.md')
s=p.read_text();marker='### 稳定训练分阶段计时、CUDA Graph 与数值差异定位（2026-10-14）';head,section=s.split(marker,1)
paragraph='''数据加载的暴露等待有长尾：两轮 auto 的中位数约 0.155–0.158 ms，但第一次在 step 210/221 为 300.89/166.95 ms，第二次 step 210 为 64.09 ms。上述均不是新编译步骤，全部保留在均值、占比和吞吐中；没有用删除慢步骤的方法提高加速比。分布及原始位置见 summary.json 的 data_wait_distribution。准备阶段均值也包含 CPU 暴露等待与通信依赖，纯 CUDA kernel 时间另见 Nsight。

'''
section=section.replace('合并稳定窗口输入吞吐：',paragraph+'合并稳定窗口输入吞吐：')
lines=section.splitlines();formatted=[]
for line in lines:
 if line.startswith('- ') and formatted and formatted[-1] and not formatted[-1].startswith('- '):formatted.append('')
 formatted.append(line)
p.write_text(head+marker+'\n'.join(formatted)+'\n')
