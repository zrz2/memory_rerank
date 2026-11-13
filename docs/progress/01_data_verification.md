# 阶段 1：数据核验

已完成两套公开候选的数据与授权初筛。LongMemEval_S 和 oracle 均为 500 个完整 JSON 实例，S 文件通过 LFS SHA256 校验；LoCoMo10 已检查对话 ID 与 evidence 映射，保留原始异常清单。

用户已指出旧「记忆调研」中的 GA 笔记／演示不可信。本项目不使用其代码、样例、成绩或结论，新的测试和实现独立完成。

尚未完成公开集正式模型实验。本阶段没有研究性能结论。下一步建立真实 embedding 的小规模运行链路，并保存原始排名与日志。

## 基线实现准备

已安装隔离环境：Python 3.12、torch 2.6.0 CPU wheel、sentence-transformers 4.1.0、transformers 4.51.3、numpy 1.26.4、scikit-learn 1.6.1。固定本地 multilingual-e5-small revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`，首次编码 smoke test 输出 384 维向量，无外部 API 调用。
