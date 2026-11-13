# 第一版数据约定

日期：2026-11-13。原始公开数据保留原样，处理结果放入 `data/processed`，不提交大型原始数据。

## 一个实验任务

`task_id`、`memories`、`queries`。每条 memory 有 id、content、created_at、last_accessed_at、source；可选 importance、importance_source、task_tags、valid_from、valid_to、supersedes。

每条 query 有 id、text、query_time；可选 target_time、task_tags；评价标签为 relevance_grades、obsolete_ids、required_evidence_sets。标签只进入评价函数，不输入 embedding 或排序器。

## 时间

自建数据明确使用 Asia/Shanghai 的 +08:00。公开数据若没有时区，只为一致运算假定 UTC，并明确记录该转换，不能把它说成原始真实时区。所有运行均先限制 created_at <= query_time。

created_at 表示可见时间，last_accessed_at 表示访问时间，valid_from/to 表示事实适用时间。有效区间为左闭右开。历史查询根据 target_time 判断有效性，但仍不能读到 query_time 后写入的数据。

## 重要性与任务标签

合成诊断题的重要性和标签是设计者给定的测试输入，不是 LLM 生成结果，也未经过独立人工复核。使用它们的结果只能表明在给定元数据条件下程序的表现。公开数据没有 importance 时使用常数 5，并标为 missing_constant_5；这时 GA 三项评分中的 importance 对所有候选相同，不能称为完整 GA 检索复现。

## 第一轮范围

本阶段不训练权重，不用运行结果挑选样本，不把小样本 smoke 实验称为正式论文结果。下一阶段建立独立开发／测试集并补充重要性模型和强重排器。
