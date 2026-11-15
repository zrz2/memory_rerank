# Mem0 2.2.1 复现细节（仅最新版）

记录日期：2026-11-15。正文只记录 `mem0ai==2.2.1`；论文同期 `v0.1.94` 的四操作机制实测单独放在附录。本文不是论文 LOCOMO / Table 1 的复现，而是**机制级复现**：验证当前版本在“地址变更 + 历史查询”下实际执行什么操作、检索返回什么、旧事实是否排前。

证据文件：

- 代码：`experiments/mem0/code/run_smoke.py`、`run_project.py`、`mem0_adapter.py`
- 结果：`experiments/mem0/results/smoke.json`、`results/project.json`
- 运行库：`experiments/mem0/runs/smoke/`、`runs/project/`
- LLM 缓存与账本：`experiments/mem0/runs/api/smoke/`

## 1. 复现问题

1. 新事实进来时，2.2.1 默认执行 `ADD`，还是 `UPDATE` / `DELETE` / `NOOP`？
2. 同一用户先住北京、后搬上海，问“现在住哪”时旧事实排第几？
3. 检索分数由哪些项组成？有没有近因、有效期或任务相关性？
4. 一次运行实际调用多少次 LLM、消耗多少 token、延迟多少？

### 结果速览

| 运行/查询 | gold | gold rank | MRR | 第 1 名 |
|---|---|---:|---:|---|
| smoke：现在住哪 | 上海（人工解释） | 2 | 0.5 | 北京 0.6484 |
| smoke：忌口 | 花生过敏 | 1 | 1.0 | 花生过敏 0.6152 |
| `address_current` | `a_new` | 2 | 0.5 | `a_old` 0.6122（已过期） |
| `address_history` | `h_old` | 1 | 1.0 | `h_old` 0.7854（文本相似，非时间推理） |

## 2. 环境与版本

| 项 | 值 |
|---|---|
| mem0ai | 2.2.1 |
| qdrant-client | 1.20.0 |
| sentence-transformers | 4.1.0 |
| Python | 3.12 |
| 嵌入模型 | 本地 BGE-M3，revision `5617a9f`，1024 维，CPU |
| 向量库 | Qdrant 本地文件，`on_disk=True` |
| LLM 配置 | DeepSeek 官方端点，`deepseek-chat`，temperature 0；适配器 `max_tokens=1600`（`configs/llm.json` 为 512） |
| 实际返回模型 | `deepseek-flash`（账本 `returned_model`） |
| 运行模式 | HuggingFace 离线；`MEM0_TELEMETRY=False` |

## 3. 配置与适配

两个入口共用同一套配置，只有 `collection_name`（`mem0_smoke` / `mem0_project`）与运行目录不同：

```python
config = {
    "embedder": {
        "provider": "huggingface",
        "config": {"model": MODEL, "embedding_dims": 1024,
                   "model_kwargs": {"device": "cpu", "local_files_only": True}},
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {"collection_name": "mem0_smoke",
                   "embedding_model_dims": 1024,
                   "path": str(RUN / "qdrant"), "on_disk": True},
    },
    "llm": {"provider": "openai", "config": {"api_key": "unused"}},
    "history_db_path": str(RUN / "history.db"),
    "version": "v1.1",
}
memory = Memory.from_config(config)
memory.llm = AuditedDeepSeek({})
```

关键点：

- `version="v1.1"` 是 Mem0 配置版本，不是包版本。
- LLM 先占位，再替换为 `AuditedDeepSeek`，保留 Mem0 原始提示词，只换传输层。
- 适配器设置 `max_tokens=1600`，不是全局配置里的 512。
- 适配器拒绝工具调用；2.2.1 默认 ADD-only 管线不使用四操作工具调用。
- 适配器的 LLM 缓存键包含 `messages`、`response_format`、LLM 配置、`max_tokens`，**不含 `user_id`**；相同文本会复用调用。
- 预算约束：共享 5 元上限、最多 8 次调用；请求失败不自动付费重试。

## 4. 代码、数据与运行命令

| 内容 | 路径 |
|---|---|
| 最小示例入口 | `experiments/mem0/code/run_smoke.py` |
| 诊断集入口 | `experiments/mem0/code/run_project.py` |
| LLM 适配器 | `experiments/mem0/code/mem0_adapter.py` |
| 诊断任务与人工标签 | `data/diagnostic/tasks_v0.json` |
| 最小示例结果 | `experiments/mem0/results/smoke.json` |
| 诊断集结果 | `experiments/mem0/results/project.json` |

```bash
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe experiments/mem0/code/run_smoke.py
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe experiments/mem0/code/run_project.py
```

## 5. 运行 A：3 轮英文最小示例

### 5.1 写入结果

| 轮次 | 输入 | 抽取出的事实 | 操作 |
|---|---|---|---|
| 1 | I moved to Beijing in March 2025 for a new job. I love it here. | User moved to Beijing in March 2025 for a new job and loves living there | ADD |
| 2 | By the way, I am allergic to peanuts. Please remember that. | User is allergic to peanuts | ADD |
| 3 | Actually, the Beijing move was temporary. I relocated to Shanghai in June 2025 and I live there now. | User's move to Beijing in March 2025 was temporary; they relocated to Shanghai in June 2025 and currently live there | ADD |

3 轮全部 `ADD`，旧北京记录没有被覆盖或删除。第 3 轮抽取出的事实虽然写了“relocated to Shanghai”，但北京条目仍留在库中。

### 5.2 检索结果

查询 1：`Where does the user live now?`

| 排名 | 记忆 | 分数 |
|---:|---|---:|
| 1 | User moved to Beijing in March 2025 for a new job and loves living there | 0.6484 |
| 2 | User's move to Beijing in March 2025 was temporary; they relocated to Shanghai in June 2025 and currently live there | 0.6304 |
| 3 | User is allergic to peanuts | 0.5712 |

正确当前地址是上海，但旧北京排第 1；按“上海为 gold”计算，gold rank = 2，MRR = 0.5。

查询 2：`Which food should the user avoid?`

| 排名 | 记忆 | 分数 |
|---:|---|---:|
| 1 | User is allergic to peanuts | 0.6152 |
| 2 | User moved to Beijing in March 2025 for a new job and loves living there | 0.3272 |
| 3 | User's move to Beijing in March 2025 was temporary; they relocated to Shanghai in June 2025 and currently live there | 0.3006 |

过敏事实排第 1；gold rank = 1，MRR = 1.0。这个例子说明语义检索在直接匹配时有效，但不解决新旧冲突。

## 6. 运行 B：2 个中文诊断任务

### 6.1 任务与写入

`address_current` 写入 3 条：

| 来源 ID | 输入 | 抽取出的事实 | 操作 | 返回 ID 前缀 |
|---|---|---|---|---:|
| `a_old` | 10月，我的收货地址是杭州。 | User's delivery address for October is in Hangzhou, China | ADD | `955d2fc9` |
| `a_new` | 11月10日起，我的收货地址改为上海。 | User's delivery address changed to Shanghai starting November 10, 2026, replacing the previous Hangzhou address used in October | ADD | `b6ebeedb` |
| `a_story` | 我以前住在杭州，那里有很多西湖边的回忆。 | User previously lived in Hangzhou, China, and has many memories of the West Lake (西湖) area there | ADD | `46fc6bed` |

`address_history` 写入 2 条：

| 来源 ID | 输入 | 抽取出的事实 | 操作 | 返回 ID 前缀 |
|---|---|---|---|---:|
| `h_old` | 10月，我的收货地址是杭州。 | User's delivery address for October is in Hangzhou, China | ADD | `b477f5fe` |
| `h_new` | 11月10日起，我的收货地址改为上海。 | User's delivery address changed to Shanghai starting November 10, 2026, replacing the previous Hangzhou address used in October | ADD | `7c3d0c85` |

5 次 `add` 全部返回 `ADD`。没有 `UPDATE`、`DELETE` 或 `NOOP`。

### 6.2 元数据与时间字段

- 每条记忆写入时带 `metadata={"source_memory_id": ..., "created_at": ...}`。
- 输出里 `metadata` 只保留 `source_memory_id`；`created_at` 被放到顶层，值等于传入的事件时间（如 `2026-10-01T09:00:00+08:00`），**不是实际写入时间 2026-11-13**。
- 这说明 2.2.1 本地 OSS 的时间要靠元数据携带；不能把顶层 `created_at` 当成“入库时间”。
- 结果回连原始记忆必须用 `metadata.source_memory_id`，不能靠文本匹配（抽取会把中文改写成英文）。

### 6.3 `address_current`：问“今天的快递寄到哪里？”

查询时点 `2026-11-13T10:00:00+08:00`，gold = `a_new`，`a_old` 标为过期。

| 排名 | 来源 ID | 记忆 | 分数 |
|---:|---|---|---:|
| 1 | `a_old` | User's delivery address for October is in Hangzhou, China | 0.6122 |
| 2 | `a_new` | User's delivery address changed to Shanghai starting November 10, 2026... | 0.5710 |
| 3 | `a_story` | User previously lived in Hangzhou, China... | 0.4279 |

结果：**过时的杭州地址排第 1**；gold rank = 2，MRR = 0.5，`stale_ids_in_top5 = ["a_old"]`。

### 6.4 `address_history`：问“10月的收货地址是什么？”

查询时点 `2026-11-13T10:00:00+08:00`，`target_time = 2026-10-20T10:00:00+08:00`，gold = `h_old`。

| 排名 | 来源 ID | 记忆 | 分数 |
|---:|---|---|---:|
| 1 | `h_old` | User's delivery address for October is in Hangzhou, China | 0.7854 |
| 2 | `h_new` | User's delivery address changed to Shanghai starting November 10, 2026... | 0.6301 |

结果：gold rank = 1，MRR = 1.0，`stale_ids_in_top5 = ["h_new"]`。

**但不能把这解释为时间推理成功**：2.2.1 的 `search` 没有 `target_time` 参数，查询文本里含“10月”，只是文本相似把杭州条目排前。Mem0 不知道“10 月 20 日时上海还没生效”。

### 6.5 分数构成

检索调用参数为 `top_k=5`、`explain=True`；内部先按 `max(top_k×4, 60) = 60` 条做候选，再融合。`score_details` 在两条查询、五条结果中完全一致：

```text
semantic_score = final_score
bm25_score     = 0.0
entity_boost   = 0.0
threshold      = 0.1
```

即当前环境是 **dense-only**：没有 spaCy / fastembed，BM25 与实体增益为 0。检索分数里没有访问近因、有效期、任务相关性，也没有重排阶段。

## 7. 调用次数、token、延迟与成本

### 7.1 调用次数

- `add` 调用：3（smoke）+ 5（project）= **8 次**。
- 其中 project 有两段文本与另一任务重复（“10月……”和“11月10日起……”各出现两次），适配器缓存键不含 `user_id`，因此复用。
- 实际 LLM 调用：**6 次**（账本 `runs/api/smoke/ledger.json` 有 6 条记录）。
- 检索不调用 LLM；`search` 只做本地嵌入 + Qdrant 检索。
- 嵌入调用未单独插桩计数；按代码路径每次 `add` / `search` 至少一次本地 BGE-M3 编码，但**不能把估算当账本实测**。

### 7.2 token

| 项 | 数值 |
|---|---:|
| prompt tokens | 47,353 |
| completion tokens | 287 |
| total tokens | 47,640 |

第 2–6 次调用命中 prompt 缓存（`prompt_cache_hit_tokens` 分别为 7552、7680、7680、7552、7552），但输入仍需按未缓存上界估算成本。

### 7.3 单次 LLM 延迟

| 缓存键前缀 | prompt | completion | total | 延迟（秒） |
|---|---:|---:|---:|---:|
| `0da4cbff` | 7,854 | 42 | 7,896 | 0.649 |
| `3f6bc533` | 7,899 | 30 | 7,929 | 1.391 |
| `d7dc844d` | 7,940 | 60 | 8,000 | 1.169 |
| `09078eee` | 7,845 | 37 | 7,882 | 1.068 |
| `6e0545a6` | 7,883 | 59 | 7,942 | 1.062 |
| `d02778ae` | 7,932 | 59 | 7,991 | 0.625 |

合计 5.964 秒，均值 0.994 秒，最小 0.625 秒，最大 1.391 秒。**这是 LLM 单次调用延迟，不是端到端延迟**；嵌入和 Qdrant 检索未计入。

### 7.4 成本

- 账面保守预留：**2.54643 元**（按适配器公式逐次累加，不是扣款）。
- 按核验当日 Flash 高峰价、全部输入按未缓存估算：`47,353 × 2 / 1e6 + 287 × 8 / 1e6` ≈ **0.0970 元**。
- 实际扣款以平台账单为准；本文没有保存账单截图，因此不宣称精确实际费用。

### 7.5 模型身份

配置写 `deepseek-chat`，账本 6 次调用的 `returned_model` 都是 `deepseek-flash`。复现记录必须同时保留“配置模型”和“实际返回模型”。

## 8. 结果说明

1. **2.2.1 默认只增。** 8 次 `add` 全部 `ADD`；没有执行论文里的 `UPDATE` / `DELETE` / `NOOP`。源码中经典四操作提示仍存在，但默认 V3 管线不调用。
2. **只增策略会让过时事实排前。** `address_current` 中旧杭州 0.6122 排第 1，新上海 0.5710 排第 2；这不是提示词写错，而是检索分数只看向量相似度。
3. **没有时效与任务信号。** `score_details` 只有 semantic，BM25 与实体增益为 0；没有 recency、`valid_from/to`、`task_tags`。
4. **历史查询的第 1 名不能算时间推理。** 2.2.1 没有 `target_time`，杭州排前是因为查询文本含“10月”。
5. **检索不是重排器。** 没有候选池、加权融合或证据级排名；论文的评测是 answer-level，检索只返回记忆条目，不能直接当本课题的排序基线。
6. **中文抽取会改写为英文。** 回连原始记忆只能靠 `source_memory_id`，不能靠文本匹配。

### 8.1 源码核对（2.2.1）

| 观察 | 证据 |
|---|---|
| 默认 `add` 走 V3 批量管线 | `memory/main.py:918`、`:944` |
| V3 提示固定“唯一操作是 ADD” | `configs/prompts.py:472` |
| 四操作提示仍存在，但默认路径不调用 | `configs/prompts.py:176–290`、`:406` |
| 检索组合分 = 语义 + BM25 + 实体增益 | `utils/scoring.py:57,72,118` |

源码路径相对 `.venv/Lib/site-packages/mem0/`。本次运行没有修改 Mem0 本体，只替换 LLM 传输层。

## 9. 不能得出的结论

- 不是论文 LOCOMO / Table 1 的复现：模型（DeepSeek 而非 GPT-4o-mini）、提示词、数据、指标都不同，数字不可横比。
- 一次 3 轮示例 + 两个诊断任务不能代表 Mem0 整体性能。
- 不能把 `address_history` 的 MRR=1.0 解释为时间推理能力；它没有 `target_time`。
- 不能把“8 次 add”与“6 次付费调用”混为一谈：缓存键不含 `user_id`，重复文本复用。
- 不能宣称完整混合检索：缺 spaCy / fastembed，BM25 与实体增益为 0。
- 不能宣称端到端延迟或完整成本：只记录了 LLM 单次延迟与保守预留。

## 10. 复现步骤

```bash
# 1. 最小示例
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe experiments/mem0/code/run_smoke.py

# 2. 诊断集
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe experiments/mem0/code/run_project.py
```

干净重跑：

```bash
rm -rf experiments/mem0/runs/smoke/qdrant experiments/mem0/runs/smoke/history.db
rm -rf experiments/mem0/runs/project/qdrant experiments/mem0/runs/project/history.db
```

LLM 缓存保留时重跑不重复计费；缓存位于 `experiments/mem0/runs/api/smoke/`。需要 `DEEPSEEK_API_KEY`；嵌入模型在 `.cache/models/bge-m3`。

## 11. 验收清单

- [x] 固定 `mem0ai==2.2.1`、qdrant-client 1.20.0、sentence-transformers 4.1.0。
- [x] 固定本地 BGE-M3、1024 维、CPU。
- [x] 记录 8 次 add 的输入、抽取结果、操作与返回 ID。
- [x] 记录 2 个诊断任务的逐条排名与分数。
- [x] 记录 gold rank、MRR、过时条目。
- [x] 记录 prompt / completion token、6 次 LLM 调用、单次延迟。
- [x] 记录账面预留与成本上界，不冒充实际扣款。
- [x] 明确 dense-only、无 recency/有效期/任务项、无重排。
- [x] 明确历史查询第 1 名不构成时间推理证据。

## 12. 附录：论文同期 v0.1.94 四操作机制实测

正文记录的是 `mem0ai==2.2.1`，它默认只 `ADD`。为展示论文 §2.1 / Appendix B Algorithm 1 的四操作设计，另用论文原文给出的官方仓库运行：

| 项 | 值 |
|---|---|
| 论文代码入口 | 正文第 29 行：`Code can be found at: https://mem0.ai/research` |
| 官方仓库 | <https://github.com/mem0ai/mem0> |
| tag | `v0.1.94` |
| commit | `07ddd7cb4bd67962cf9a988d7b5c3f3920fad2d4` |
| commit 日期 | 2025-04-26 16:22:20 +0530 |
| 选择依据 | arXiv v1（2025-04-28）前最后一个官方 tag；v0.1.95 发布于 2025-04-30 |

入口：`experiments/mem0/paper_mechanism/run_four_ops.py`；结果：`experiments/mem0/results/paper_mechanism.json`。

### 12.1 逐轮决策（5 轮英文对话）

| 轮次 | 输入 | `add()` 返回的决策 | 原始 LLM 更新响应中的其他项 |
|---:|---|---|---|
| 1 | I moved to Beijing in March 2025 and work at ByteDance. | ADD 北京；ADD 字节 | — |
| 2 | By the way, I am allergic to peanuts. | ADD 花生过敏 | 旧北京、旧字节 NONE |
| 3 | Actually, I relocated to Shanghai in June 2025; I live there now. | UPDATE 北京→上海；ADD 当前住上海 | 字节、过敏 NONE |
| 4 | Correction: I never joined ByteDance. | DELETE 字节；ADD 从未入职 | 上海两条、过敏 NONE |
| 5 | I am still allergic to peanuts. | 空（全部 NONE） | 四条全部 NONE |

`NONE` 只在原始响应里出现，`add()` 不把它放进返回值；结果 JSON 的 `llm_responses` 保留原文。

### 12.2 最终记忆与检索

第 4 轮后库中四条：`Never joined ByteDance`、`Allergic to peanuts`、`Relocated to Shanghai in June 2025`、`Currently lives in Shanghai`。

| 查询 | 第 1 名 | 分数 |
|---|---|---:|
| Where does the user live now? | Currently lives in Shanghai | 0.7205 |
| Where did the user live before? | Currently lives in Shanghai | 0.6149 |
| Where does the user work? | Currently lives in Shanghai（无关） | 0.4807 |
| Which food should the user avoid? | Allergic to peanuts | 0.5066 |

**UPDATE 覆盖旧值**：北京被改写成上海，“以前住哪”也搜不到北京；**DELETE 物理删除**：字节条目消失，“在哪工作”无对应证据。这与论文“DELETE 会丢历史”的判断一致。

### 12.3 调用与成本

| 项 | 值 |
|---|---:|
| LLM 调用 | 10 次（5 轮 × 抽取 + 更新） |
| prompt tokens | 11,390 |
| completion tokens | 682 |
| 账面预留 | 1.23274 元 |
| 实际返回模型 | `deepseek-flash` |

### 12.4 源码核对与差异

| 观察 | 证据 |
|---|---|
| 抽取后检索旧记忆 `limit=5` | `mem0/memory/main.py:244` |
| 调用四操作提示 | `mem0/memory/main.py:261` |
| 处理 UPDATE / DELETE / NONE | `mem0/memory/main.py:302`、`:317`、`:326` |
| 提示词列出四操作 | `mem0/configs/prompts.py:62-70` |

**差异**：论文 §2.1 写更新阶段检索 `s=10`，v0.1.94 代码是 `limit=5`。本附录记录代码实际行为，不把 5 说成 10。

### 12.5 不能宣称

- 不是 LOCOMO / Table 1 复现；模型、数据、指标均不同。
- 一次 5 轮机制演示不代表 Mem0 整体性能。
- 该结果是“论文同期版本”的行为，不能直接等同于 2.2.1 的当前默认行为。
