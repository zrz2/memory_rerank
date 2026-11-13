# 实验协议 v0：仅用于流程验证

日期：2026-11-13。第一轮公开样本 ID 在编码前冻结，按源文件每类第一道非 abstention 题选取，没有根据输出挑题。

## 不变项

- 模型：multilingual-e5-small，revision 614241f622f53c4eeff9890bdc4f31cfecc418b3，CPU，batch=8，max_seq_length=256，归一化向量。
- Query 使用 `query: `，记忆使用 `passage: `；不把 gold labels、正确答案或证据 ID 输入模型。
- 同一问题的所有方法使用相同完整可见历史，未额外进行语义 Top-N 截断。
- 各因素在本题可见候选内 min-max；等值设 0.5；分数并列按 memory_id。
- 评分只读取 query_time、可选 target_time/task_tags；qrels 只进入评价函数。
- 使用离线固定访问状态，不在评估中更新 last_accessed。

## 两类数据不可混称

诊断题：10 个手工任务，其中 1 个无充分证据。task 标签、importance、有效时间均是人为设定的测试输入，未独立人工复核。有效性 oracle 用于检查机制，不是自动方法。

公开题：LongMemEval_S cleaned 的 6 个固定问题。保留完整历史轮次，基于 has_answer 作二元 evidence 标签。last_accessed 初始化为会话时间；importance 统一为常数 5。因此公开集 GA-paper-score 与 semantic_recency 应保持相同排名，不能宣称完整 GA 检索性能。

## 长文本与时间

公开集每轮对话按 200 token 窗口、160 token 步长切块，窗口相似度取最大值回聚到轮次。该策略完整覆盖原轮次，不是官方脚本默认策略；长文本更多窗口也可能带来最大值偏差，正式协议需对照。

原日期未指定时区，统一赋予 UTC 以保持相对时间；不声称其原始时区是 UTC。所有候选均需在 query_time 前可见。

## 指标

K=1/3/5/10（诊断为1/3/5）；报告 Recall@K、截断 MRR@K、NDCG@K、完整证据命中率。无 gold 的题不参与排名均值，也不将“检索了无关记录”等同于完整拒答系统表现。oracle 有效性可能返回少于 K 条；保留实际返回条数。

## 不得作出的结论

- 不能凭六题或人工诊断集声称统计显著、达到 SOTA 或优于 GA 原论文。
- 不能将固定重要性当作模型打分。
- 不能将正确元数据的过滤收益都归为重排模型学到的能力。
- 不能将未标 has_answer 的所有其他轮次认定为完全无信息，公开证据标注可能不穷尽复述。

## 下一版必须补充

独立开发／测试划分、真正的 importance 评分与缓存、强 Cross-Encoder、query+task_state 对照、元数据自动抽取或明确 oracle 条件、更多样本及按任务的不确定性估计。
