# 阶段 2：排序核心与诊断闭环

本阶段不使用旧 `记忆调研/06_demo` 的代码、数据或结果。新代码独立放在 `src/memory_rerank`。

已完成：

- 论文式 GA-paper 分数：访问近因性、语义相似度、importance，各自归一化后等权相加。
- 纯语义、语义+近因性、带任务标签诊断、带有效时间诊断的对照方法。
- 明确的时间可见性、有效区间、访问状态隔离和指标实现。
- 固定 multilingual-e5-small，首次本地编码 smoke test 输出 384 维向量，未调用外部 API。
- 9 个排序单元测试通过；通过后再运行 10 个任务诊断。

当前 `tasks_v0` 是人工编写的流程诊断题，标签属于测试输入，不是从公开数据自动抽取的；结果不用于论文性能结论。运行命令：

```powershell
E:/毕业论文/memory-rerank/.venv/Scripts/python.exe scripts/run_diagnostic.py
```

运行结果保存到 `runs/diagnostic_v0.json`，包含每题各方法的分项分数、排名和指标。

需要后续修正的问题：

- 第一版任务数量和措辞尚未经过第二位人工复核。
- GA-paper 的 importance 暂为人工诊断分数，公开集没有对应字段时不能声称完整复现。
- 任务标签和有效时间的 oracle 版本用于机制验证；自动抽取版本要单独实验。
- 目前只使用单一 embedding 模型，尚未加入强 Cross-Encoder 对照。
