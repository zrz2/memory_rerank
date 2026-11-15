# 智能体记忆检索重排序

课题：**融合语义、时效与任务相关性的智能体记忆检索重排序方法研究**。

远程仓库：<https://github.com/zrz2/memory_rerank>

## 课题目标

设计并验证一个**离线、可复现的智能体记忆系统**，覆盖记忆的完整生命周期：

```text
多轮对话 / 文档
   → L1 写入（抽取、来源、时间字段）
   → L2 维护（valid_from/to、supersedes、冲突策略）
   → L3 召回（BGE-M3 top-N；同一候选池）
   → L4 重排（有效性约束 + 语义/近因/重要性/任务相关性）
   → top-K 证据 → L5 指标 / 可选下游回答
```

系统输入为多轮对话、查询时点 `t_q`、任务状态 `z` 和查询 `q`；输出 top-K 记忆证据。重点解决两类失败：

1. 语义相关但已失效，例如旧地址、已取消的计划；
2. 语义不明显但影响当前决策，例如膝伤对酒店电梯需求的影响。

三个边界：

- **系统完整，实验分层。** 存储、更新、检索、重排都要设计，但比较时一次只改变目标层；用“理想元数据 vs 自动管线”两层实验分别验证排序方法与系统误差。
- **重排是最后一步评分，不是全部。** 上游表示、更新策略和候选构造决定重排上限；上游删掉历史，重排无法恢复。
- **两种时间分开。** 访问近因性回答“多久没被用到”；事实有效性回答“在 `t_q` 是否成立”。`valid_from/valid_to` 是硬约束，不是普通加分项。

本阶段（至 2026-12-21）目标：完成 10 段多轮玩具任务与三方案 demo（仅语义 / 语义+时效 / 语义+时效+任务标签），输出 Recall@K、MRR、NDCG@K、耗时与模型调用次数，并完成一个公开数据集的可行性检查。

明确不做：完整产品、实时上下文分页、完整复现 Mem0 / Generative Agents、训练大模型，以及用 10 题宣称统计显著优势。

## 笔记位置

| 笔记 | 位置 |
|---|---|
| Generative Agents（三因子检索基线） | [`docs/papers/generative_agents.md`](docs/papers/generative_agents.md) |
| MemGPT（当前状态 vs 原始证据） | [`docs/papers/memgpt.md`](docs/papers/memgpt.md) |
| Mem0（维护与冲突处理） | [`docs/papers/mem0.md`](docs/papers/mem0.md) |

## Mem0 复现位置

只记录 `mem0ai==2.2.1` 最新版：

| 内容 | 位置 |
|---|---|
| 复现细节与逐次结果 | [`experiments/mem0/REPRODUCTION.md`](experiments/mem0/REPRODUCTION.md) |
| 论文同期 v0.1.94 四操作实测 | [`experiments/mem0/paper_mechanism/`](experiments/mem0/paper_mechanism/README.md) |
| 最小示例入口 | [`experiments/mem0/code/run_smoke.py`](experiments/mem0/code/run_smoke.py) |
| 诊断集入口 | [`experiments/mem0/code/run_project.py`](experiments/mem0/code/run_project.py) |
| LLM 适配器（预算、缓存、账本） | [`experiments/mem0/code/mem0_adapter.py`](experiments/mem0/code/mem0_adapter.py) |
| 结果 | [`experiments/mem0/results/`](experiments/mem0/results/) |
| 目录说明与运行命令 | [`experiments/mem0/README.md`](experiments/mem0/README.md) |

## 仓库结构

```text
docs/
  implementation_plan.md      实施方案与阶段边界
  NEXT.md                     唯一进度台账
  papers/                     三篇核心笔记
experiments/mem0/             Mem0 复现代码、结果与详细记录
data/diagnostic/tasks_v0.json 共用的诊断任务与人工标签
src/memory_rerank/            嵌入、打分、指标与 LLM 调用
scripts/                      数据检查、运行与汇总脚本
tests/                        公式、时间与指标测试
```

## 运行

当前统一配置：DeepSeek `deepseek-chat` + 本地 `BAAI/bge-m3` dense embedding。详见 [`docs/environment.md`](docs/environment.md)。

```bash
# 测试
./.venv/Scripts/python.exe -m pytest -q

# Mem0 2.2.1 最小示例
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe experiments/mem0/code/run_smoke.py

# Mem0 2.2.1 诊断集
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe experiments/mem0/code/run_project.py
```

需要 `.env` 中的 `DEEPSEEK_API_KEY`（不提交 Git）；嵌入模型在 `.cache/models/bge-m3`。`runs/` 与 `experiments/mem0/runs/` 是运行时向量库、缓存和账本，已被 `.gitignore` 排除，可按 `REPRODUCTION.md` 重建。

## 当前状态

- 已完成：三篇核心笔记；Mem0 2.2.1 的最小示例与两个诊断任务复现。
- 已有：10 个手工诊断任务、纯语义/近因性/GA 式打分与 Recall/MRR/NDCG 代码。
- 尚未完成：10 段多轮玩具任务、L2 维护模块、正式公开基准实验、Cross-Encoder 对照与下游回答评测。
- 现有结果仅用于流程验证与失败模式发现，不是正式论文结论；数字不可与 Mem0 论文 Table 1 横比。
