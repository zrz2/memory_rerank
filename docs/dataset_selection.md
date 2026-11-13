# 数据集初筛与实际核验

日期：2026-11-13。证据是已下载的原始文件、官方 README、许可证和 `reports/dataset_audit.json`；不是仅根据论文介绍猜测。

## 选择

**主候选：LongMemEval_S cleaned。** 已完整下载并按源端 LFS SHA256 校验，500 个问题，包含 78 个 knowledge-update、133 个 temporal-reasoning；共 30 个 abstention 问题（由 ID 后缀识别，与六类问题类别交叉）。官方提供 `answer_session_ids` 和部分轮次的 `has_answer`，能支持会话／轮次层面的检索诊断。

**补充候选：LoCoMo10。** 实际读取到 10 个长对话、1986 个 QA、5882 个对话轮次。4 题无 evidence，9 题的原始 evidence 字符串不能直接精确匹配轮次 ID，可能包括格式问题和不存在的 ID。当前不自动修正，不把这些问题悄悄当作标准金标。另有 910 个带图像 URL 的轮次，纯文本实验须说明只用了文本／图注，并不具备图像理解能力。

## 官方版本和授权

- LongMemEval 代码：`xiaowu0162/LongMemEval`，commit `9e0b455f4ef0e2ab8f2e582289761153549043fc`，MIT。
- LongMemEval 数据：`xiaowu0162/longmemeval-cleaned`，revision `98d7416c24c778c2fee6e6f3006e7a073259d48f`，数据卡标记 MIT；采用 cleaned 版本，不能与早期未经清理的成绩直接混比。
- LoCoMo：`snap-research/locomo`，commit `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`。根许可证是 CC BY-NC 4.0，需署名、遵守非商业限制并说明修改。原始数据保留本地，不自动发布。
- 原始来源和锁定版本位于 `configs/dataset_sources.json`。

## 文件核验

LongMemEval_S 原始大小 277383467 字节，500 个实例。会话 ID、时间、正文长度对齐；证据会话均能找到；非拒答题均有标记轮次。oracle 文件也已下载，但只用于理解结构，**不能把 oracle-only 的检索成绩作为完整干扰环境结果**。

第一次普通下载发生无报错截断，JSON 校验失败；已用逐段 HTTP Range、长度和 SHA256 核验重新下载，坏文件已由验证后的文件替换。下载脚本保留在仓库。

## 第一轮实验边界

先从 LongMemEval_S 每个问题类别按源文件顺序选第一个非拒答问题，共 6 题，冻结 ID。使用该题全部历史轮次作为候选，按官方 has_answer 标记检索证据，不输入答案或标记给模型。6 题仅用于端到端流程验证，不是正式测试集，也不据此调参或声称性能提高。

公开数据没有任务标签、原始 GA 的访问历史或重要性评分。第一轮只能比较语义与创建时间代理的 recency；last_accessed 初始化为 created，importance 暂设常数并明确标记。不能把这一结果称为完整 GA 模型复现。

后续需验证：标记轮次与未标记复述的关系、证据切块粒度、正式划分、自动重要性评分、强重排器、下游官方 QA judge。当前未调用任何付费 judge。

## 官方入口

- https://github.com/xiaowu0162/LongMemEval
- https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
- https://github.com/snap-research/locomo

源码 README 当前还提示 LongMemEval-V2；它是后续需要单独核验的更新，不在本轮与 2024 论文数据混用。
