# LongMemEval：先看能不能用于本课题

状态：已核验官方 README、数据卡、授权和下载数据；这是数据可用性笔记，不冒充全文精读。

- 本地为 cleaned S 版本，500 题，原文件 277383467 字节，SHA256 已验证。
- 存在知识更新、时间推理等类别，适合本课题的时间方向；不等于已提供显式任务状态标签。
- 有会话级 answer_session_ids 和轮次级 has_answer，可建立对应粒度检索诊断。
- 第一轮冻结六类各第一道非拒答题，使用完整历史，做流程验证；不是正式测试划分。
- oracle 文件只含证据会话，不能替代带干扰历史的主实验。
- 官方 QA 评估依赖模型 judge；本轮仅做本地检索，不调用付费 API。

选择：主公开候选。正式比较前还需确认更细证据覆盖、划分、模型与评价口径。

来源：https://github.com/xiaowu0162/LongMemEval ，https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned 。锁定版本见 configs/dataset_sources.json。
