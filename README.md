# 智能体记忆检索重排序

课题：融合语义、时效与任务相关性的智能体记忆检索重排序方法研究。

## 当前范围

先复现记忆排序机制，不复现完整 Smallville。所有数据、缓存、环境和运行产物位于本项目下。原有 `../记忆调研` 资料保持原样，新项目会记录其已有成果与局限。

## 资料

- [实施方案](docs/implementation_plan.md)
- 原有 `../记忆调研` 中的 GA 演示和笔记已被用户判定为错误、不实际，本项目不引用。
- [重新整理的 GA 核心笔记](docs/papers/generative_agents.md)
- [数据集选择与实测核验](docs/dataset_selection.md)
- [进度与运行说明](docs/progress/02_baseline_smoke.md)

## 阶段

1. 项目与资料归档。
2. 数据集核验与统一数据格式。
3. 最小基线、测试和真实运行结果。

当前尚未完成正式基准实验，不预先宣称方法性能。

## 运行

在 PowerShell 中进入本目录。环境、模型、缓存和数据均位于 `E:/毕业论文/memory-rerank`。

```powershell
cd E:/毕业论文/memory-rerank
# 首次安装：
./scripts/setup.ps1
# 首次恢复数据和模型（本机已完成下载）：
./.venv/Scripts/python.exe scripts/fetch_sources.py
./.venv/Scripts/python.exe scripts/download_longmemeval.py --include-s
./.venv/Scripts/python.exe scripts/download_model.py
# 检查与运行：
./.venv/Scripts/python.exe -m pytest -q
./.venv/Scripts/python.exe scripts/run_diagnostic.py
./.venv/Scripts/python.exe scripts/run_longmemeval_smoke.py
./.venv/Scripts/python.exe scripts/summarize_runs.py
```

`runs/` 保存逐题结果，`reports/` 保存便于阅读的汇总。原始数据、模型和环境不进入 Git，下载脚本、固定版本、数据哈希和摘要进入 Git。

## 当前结果的含义

- 10 个手工诊断任务仅验证流程，未独立人工复核；oracle 标签／时间版本不是自动方法。
- LongMemEval 仅冻结六类各一道题做 smoke test，不是全 500 题结果。
- 公开集没有 importance，暂用常数，GA-paper-score 在该设置下与语义+近因性排序相同，不能冒充完整 GA 复现。
- 未进行任何付费 API 调用；未上传原始数据或推送远程仓库。
