# 论文同期 Mem0 四操作机制实测

本目录只做一件事：用**论文原文给出的官方代码仓库**，在论文 v1（2025-04-28）同期版本上，实际跑出 `ADD / UPDATE / DELETE / NONE` 四种决策。它不是 LOCOMO 复现，也不是 2.2.1 的默认行为。

## 来源与版本

| 项 | 值 |
|---|---|
| 论文 | arXiv:2504.19413v1（2025-04-28） |
| 论文代码说明 | 正文第 29 行：`Code can be found at: https://mem0.ai/research` |
| 官方仓库 | <https://github.com/mem0ai/mem0> |
| 版本 | 官方 tag `v0.1.94` |
| commit | `07ddd7cb4bd67962cf9a988d7b5c3f3920fad2d4` |
| commit 日期 | 2025-04-26 16:22:20 +0530 |
| 选择依据 | v0.1.94 是 arXiv v1 发布前最后一个官方 tag；v0.1.95 发布于 2025-04-30 |

源码核对（tag `v0.1.94`）：

- `mem0/memory/main.py:219` 调 `get_fact_retrieval_messages` 做抽取；
- `:244` 检索旧记忆 `limit=5`（论文 §2.1 写 `s=10`，代码是 5，**存在版本差异**）；
- `:261` 调 `get_update_memory_messages` 让 LLM 选四操作；
- `:302` / `:317` / `:326` 分别处理 `UPDATE` / `DELETE` / `NONE`；
- `mem0/configs/prompts.py:62-70` 明确列出 ADD/UPDATE/DELETE/NONE。

## 运行

```bash
# 一次性准备官方源码（已加入 .gitignore）
git clone --depth 1 --branch v0.1.94 --filter=blob:none --sparse \
  git@github.com:mem0ai/mem0.git experiments/mem0/paper_mechanism/vendor/mem0_repo
git -C experiments/mem0/paper_mechanism/vendor/mem0_repo sparse-checkout set mem0

# 跑四操作演示
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe experiments/mem0/paper_mechanism/run_four_ops.py
```

结果写入 `experiments/mem0/results/paper_mechanism.json`；LLM 缓存与账本在 `experiments/mem0/runs/api/paper_mechanism/`。

## 结果（5 轮对话）

| 轮次 | 输入 | 实际决策 |
|---:|---|---|
| 1 | I moved to Beijing in March 2025 and work at ByteDance. | ADD 北京；ADD 在字节工作 |
| 2 | By the way, I am allergic to peanuts. | ADD 花生过敏；旧记忆 NONE |
| 3 | Actually, I relocated to Shanghai in June 2025; I live there now. | **UPDATE** 北京→上海；ADD 当前住上海；其余 NONE |
| 4 | Correction: I never joined ByteDance. | **DELETE** 在字节工作；ADD 从未入职；其余 NONE |
| 5 | I am still allergic to peanuts. | 全部 **NONE**（API 返回空列表） |

第 4 轮后库里剩：`Never joined ByteDance`、`Allergic to peanuts`、`Relocated to Shanghai in June 2025`、`Currently lives in Shanghai`。

检索结果：

| 查询 | 第 1 名 | 分数 |
|---|---|---:|
| 现在住哪 | Currently lives in Shanghai | 0.7205 |
| 以前住哪 | Currently lives in Shanghai | 0.6149 |
| 在哪工作 | Currently lives in Shanghai（无关） | 0.4807 |
| 忌口 | Allergic to peanuts | 0.5066 |

两个关键点：

1. **UPDATE 覆盖旧值**：北京被原地改写成上海，所以“以前住哪”也搜不到北京；历史事实没有保留。
2. **DELETE 物理删除**：字节条目被删除，“在哪工作”没有对应证据；`Never joined ByteDance` 是更正事实，但没有排到第 1。

## 调用与成本

| 项 | 值 |
|---|---:|
| LLM 调用 | 10 次（5 轮 × 抽取 + 更新） |
| prompt tokens | 11,390 |
| completion tokens | 682 |
| 账面预留 | 1.23274 元 |
| 实际返回模型 | `deepseek-flash` |

## 不能宣称

- 这不是 LOCOMO / 论文 Table 1 的复现，数字不可与论文横比。
- 代码用 `limit=5`，论文写 `s=10`；本实验记录代码实际行为。
- `NONE` 不出现在 `memory.add()` 的返回结果里，只能从原始 LLM 响应看到；结果 JSON 的 `llm_responses` 保留了原文。
- 一次 5 轮演示只能说明机制如何工作，不能代表 Mem0 整体性能。
- 与 2.2.1 的对比是“论文同期版本 vs 当前版本”，不是“论文方法 vs 改进方法”。
