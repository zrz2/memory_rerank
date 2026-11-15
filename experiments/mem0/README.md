# Mem0 实验目录

所有 Mem0 复现代码、运行结果和运行时数据都放在这里，不再散落到 `scripts/`、`src/` 和 `runs/`。

## 目录结构

| 路径 | 内容 |
|---|---|
| `code/run_smoke.py` | 最小示例（3 轮英文对话） |
| `code/run_project.py` | 诊断集（2 个中文任务） |
| `code/mem0_adapter.py` | LLM 适配器：预算、调用上限、缓存、账本 |
| `results/smoke.json` | 最小示例的完整输出 |
| `results/project.json` | 诊断集的完整输出（含 gold_rank、MRR） |
| `runs/smoke/` | 最小示例的 Qdrant 向量库与 history.db |
| `runs/project/` | 诊断集的 Qdrant 向量库与 history.db |
| `runs/api/smoke/` | LLM 缓存与 `ledger.json` |
| `REPRODUCTION.md` | 2.2.1 逐次复现细节：配置、写入、逐条排名、token、延迟与成本 |

Mem0 本体不在本目录，仍是 `.venv/Lib/site-packages/mem0/`；它是项目依赖，不复制。

## 运行

在仓库根目录执行：

```bash
# 最小示例
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe experiments/mem0/code/run_smoke.py

# 诊断集
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe experiments/mem0/code/run_project.py
```

干净重跑：删除对应的 `runs/*/qdrant` 与 `history.db`；LLM 缓存命中则不重复计费。

## 数据说明（三层，别混）

1. **论文原文**：`data/raw/mem0/mem0_2504.19413.pdf`。
2. **诊断任务与人工标签**：`data/diagnostic/tasks_v0.json`。它是多个实验共用的输入，**不复制到本目录**，避免两份不一致。
3. **运行时数据**：本目录 `runs/` 下的向量库、history.db、LLM 缓存和账本；可删除后重建。

## 版本与口径

- 固定 `mem0ai==2.2.1`；默认只 `ADD`，经典四操作提示仍在源码中但默认路径不调用。
- 嵌入为本地 BGE-M3（CPU，1024 维）；LLM 为 DeepSeek。
- 缺 spaCy/fastembed 时是 dense-only，BM25 与实体增益为 0。
- 这些结果不冒充论文表 1；论文数字与本地结果不可直接横比。

论文笔记见 `docs/papers/mem0.md`；逐次复现细节见 `experiments/mem0/REPRODUCTION.md`。
