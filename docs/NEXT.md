# 进度与任务台账

**本文件是唯一的进度与安排记录。** 不再另建阶段进展、周报或 `progress/` 目录；每周状态直接更新在这里。最后更新：2026-11-15。

## 当前状态

- 课题定位已写入 `implementation_plan.md` §1：设计完整离线记忆系统（写入/维护/召回/重排/评测），用“理想元数据 vs 自动管线”两层实验分层归因；重排是最后评分，不是全部。
- 第 1 周交付完成：三篇核心笔记 + Mem0 最小示例与项目诊断集复现。
- 当前统一配置：BGE-M3（revision `5617a9f`）+ DeepSeek；早期 E5 配置已废弃，相关归档已从仓库删除。
- 尚未完成：10 段多轮玩具任务、L2 维护模块、Mem0 完整混合检索、20–50 题独立测试集、Cross-Encoder 对照、下游问答评测。

## 已完成

### 阶段 0：项目建立（2026-11-13）

- 独立项目位于 `E:/毕业论文/memory-rerank`，Git 远程为 `git@github.com:zrz2/memory_rerank.git`。
- 原有 `../记忆调研` 中的 GA 演示和笔记经用户判定错误、不实际；本项目不采信、不迁入，仅保留原文件避免误删。
- 导入实施方案、GA 核心笔记与源码核验说明。
- 本机约 15.7 GB 内存、RTX 3050 4 GB；Python 3.12。

### 阶段 1：数据核验（2026-11-13）

- LongMemEval_S cleaned 与 oracle 均为 500 个实例；S 文件通过 LFS SHA256 校验。
- LoCoMo10 已检查对话 ID 与 evidence 映射，原始异常单独记录，未静默修补。
- 当时固定 multilingual-e5-small（revision `614241f`）；后切换到 BGE-M3，旧结果不再作为当前配置成绩。

### 阶段 2：排序核心与诊断闭环（2026-11-13）

- 在 `src/memory_rerank` 独立实现：纯语义、语义+近因性、GA-paper 打分、时间可见性、有效区间、访问状态隔离、Recall/MRR/NDCG。
- 10 个人工诊断任务跑通；标签是测试输入，不是自动抽取结果。
- LongMemEval 六类各一道固定非拒答题跑通；公开集 importance 用常数 5，GA-paper 与语义+近因性排名相同，不能冒充完整 GA 复现。

### 第 1 周（11-10 – 11-16）

- 完成 Generative Agents 与 MemGPT 两篇六段式笔记。
- 跑通 Mem0 2.2.1 最小示例：3 轮后新旧住址共存；“现在住哪”旧北京事实排第 1（0.6484 vs 0.6304）。
- 跑通项目诊断集：`address_current` 过时杭州地址排第 1（0.6122 vs 0.5710，MRR=0.5）；`address_history` 旧地址排第 1，但 Mem0 无 `target_time`，属文本相似巧合。
- 记录 Mem0 默认 ADD-only、无 recency/有效期项、dense-only（缺 spaCy/fastembed）与调用成本。
- Mem0 笔记改为课题导向：突出“维护策略决定候选池、重排只能在池内工作”，三种策略对照下沉为核心，完整复现流程移入附录。
- 新增 `experiments/mem0/REPRODUCTION.md`：只记录 2.2.1 最新版的配置、8 次写入、逐条排名、6 次 LLM 调用、token、延迟与成本。

## 第 1 周交付清单

| 交付物 | 文件 |
|---|---|
| Generative Agents 笔记 | `docs/papers/generative_agents.md` |
| MemGPT 笔记 | `docs/papers/memgpt.md` |
| Mem0 运行记录（含完整复现流程） | `docs/papers/mem0.md` |
| 2.2.1 逐次复现细节 | `experiments/mem0/REPRODUCTION.md` |
| 复现代码与结果（统一目录） | `experiments/mem0/`，入口与数据说明见其 `README.md` |

## 优先继续

1. 2.2.1 已实测为只增且过时事实可排前；下一步在本课题系统里把三种维护策略（删旧 / 只增 / 标记失效）记录成同一查询下的候选与历史可查性对照表。
2. 安装 spaCy/fastembed 后重跑 Mem0，补完整混合检索（BM25 + 实体）对照；当前为 dense-only。
3. 明确任务相关性标注规范，修订诊断集：当前酒店例子把背景相关记录也给正分，Recall 与“关键约束是否找到”要分开解释。
4. 扩展独立任务并建立分组切分；当前 10 题仅 smoke，不自动变成测试集。
5. 引入可追溯 importance 模型评分；如需付费模型，先提供数量与预算。所有基线共享评分缓存。
6. 固定一个现成 Cross-Encoder，加入 q 与 q+task_state 对照。
7. 对 LongMemEval 更多实例进行正式评估前，先冻结评价协议和分组调参方式。
8. LoCoMo 的原始 evidence 异常逐条核实，显式记录排除／修正规则，不静默修补。
9. 平台五维表按实施方案推进，优先 Mem0 与一个开源 harness。

## 开放问题

- 用户尚未为 importance 或下游 QA judge 指定付费预算；当前全部本地运行。
- 没有自动任务标签抽取或自动版本判断，oracle 结果不能当作最终方法。
- 运行周期和数据规模需要基于当前实测成本决定，不承诺六题速度可线性外推到所有配置。

## 维护规则

- 进度、周计划、完成项只写本文件，不再新建周报或阶段进展文档。
- 每篇论文只维护一个笔记；数据集笔记只给实际采用的公开集。
- 结果原始文件留在 `runs/`，可读汇总留在 `reports/`，二者不当作进度记录。
- 方案与架构写在 `docs/implementation_plan.md`，数据格式写在 `docs/data_contract.md`。
