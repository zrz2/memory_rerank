# longmemeval_smoke：实际运行摘要

LongMemEval_S cleaned 六个固定非拒答问题，完整历史轮次候选；不是全基准成绩。importance 统一为 5，未调用 LLM 打分。

**这些数字只用于检查流程，不支持方法普遍有效或优于原论文的结论。MRR 在本报告为截断到 K 的 MRR@K。**

|方法|K|有效题数|Recall@K|MRR@K|NDCG@K|完整证据命中率|
|---|---:|---:|---:|---:|---:|---:|
|ga_paper_score|1|6|0.3056|0.6667|0.6667|0.0000|
|ga_paper_score|3|6|0.3056|0.6667|0.3848|0.0000|
|ga_paper_score|5|6|0.3889|0.6667|0.4288|0.1667|
|ga_paper_score|10|6|0.3889|0.6667|0.4288|0.1667|
|semantic|1|6|0.3056|0.6667|0.6667|0.0000|
|semantic|3|6|0.8056|0.8333|0.7107|0.6667|
|semantic|5|6|0.8056|0.8333|0.7107|0.6667|
|semantic|10|6|0.8056|0.8333|0.7107|0.6667|
|semantic_recency|1|6|0.3056|0.6667|0.6667|0.0000|
|semantic_recency|3|6|0.3056|0.6667|0.3848|0.0000|
|semantic_recency|5|6|0.3889|0.6667|0.4288|0.1667|
|semantic_recency|10|6|0.3889|0.6667|0.4288|0.1667|

## 运行口径

- 记录的编码状态：`{'texts': 5832, 'cache_hits': 24, 'computed': 5808, 'truncated_texts': 0, 'encode_seconds': 689.2064882000268, 'external_api_calls': 0}`。
- 原始逐题结果：`runs\longmemeval_smoke.json`。
- 结果 SHA256：`4607d6abfe1f8c3e8e2fb813d0840412e10fa0d86b64a8eebc5d25bce9013bb7`。
- 纯语义使用真实本地 multilingual-e5-small 向量，不是 TF-IDF。
- oracle 方法表示已给定正确任务标签／有效区间，不能与自动端到端方法混称。
- 未用这些样本调参，后续需独立开发集、测试集与强重排对照。
- 长轮次按 200 token 窗口、160 token 步长切块，按最大相似度合并回轮次；这是本项目约定。
- 没有实际访问日志：将 last_accessed 初始化为会话时间；未指定时区统一假定 UTC。
- GA-paper-score 的 importance 为常数，其排名应与 semantic_recency 相同；不是三项完整模型性能。
