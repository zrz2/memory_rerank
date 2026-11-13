"""Produce reviewable reports from recorded outputs, with explicit scope limitations."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def summarize(name):
    path = ROOT / "runs" / f"{name}.json"
    if not path.exists():
        print("Not finished:", name)
        return
    data = json.loads(path.read_text(encoding="utf8"))
    groups = defaultdict(list)
    for row in data["rows"]:
        groups[(row["method"], row["metric"]["k"])].append(row["metric"])
    stats = []
    for (method, k), metrics in sorted(groups.items()):
        row = {"method": method, "k": k, "answerable_queries": sum(m["recall"] is not None for m in metrics)}
        for key in ["recall", "mrr", "ndcg", "complete_evidence"]:
            values = [m[key] for m in metrics if m[key] is not None]
            row[key] = mean(values) if values else None
        stats.append(row)
    summary = {"source_run": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "scope": "pipeline_smoke_only", "model": data.get("model"), "encoder_stats": data["encoder_stats"],
               "git_head_of_run": data.get("git_head"), "git_dirty_of_run": data.get("git_dirty"),
               "metrics": stats}
    (ROOT / "reports" / f"{name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    notes = ("10 个手工诊断任务，其中 1 个无正确证据，排名指标不计入该题。importance、任务标签、有效时间均为人工测试输入。"
             if name == "diagnostic_v0" else
             "LongMemEval_S cleaned 六个固定非拒答问题，完整历史轮次候选；不是全基准成绩。importance 统一为 5，未调用 LLM 打分。")
    lines = [f"# {name}：实际运行摘要", "", notes,
             "", "**这些数字只用于检查流程，不支持方法普遍有效或优于原论文的结论。MRR 在本报告为截断到 K 的 MRR@K。**", "",
             "|方法|K|有效题数|Recall@K|MRR@K|NDCG@K|完整证据命中率|",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in stats:
        lines.append(f"|{row['method']}|{row['k']}|{row['answerable_queries']}|" + "|".join(f"{row[k]:.4f}" for k in ["recall", "mrr", "ndcg", "complete_evidence"]) + "|")
    lines += ["", "## 运行口径", "", f"- 记录的编码状态：`{data['encoder_stats']}`。",
              f"- 原始逐题结果：`{summary['source_run']}`。", f"- 结果 SHA256：`{summary['sha256']}`。",
              f"- 本次运行记录的向量模型：`{data.get('model', '未记录')}`。",
              "- oracle 方法表示已给定正确任务标签／有效区间，不能与自动端到端方法混称。",
              "- 未用这些样本调参，后续需独立开发集、测试集与强重排对照。"]
    if name != "diagnostic_v0":
        lines += ["- 长轮次按 200 token 窗口、160 token 步长切块，按最大相似度合并回轮次；这是本项目约定。",
                  "- 没有实际访问日志：将 last_accessed 初始化为会话时间；未指定时区统一假定 UTC。",
                  "- GA-paper-score 的 importance 为常数，其排名应与 semantic_recency 相同；不是三项完整模型性能。"]
    (ROOT / "reports" / f"{name}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    print(name, "summarized", len(stats), "rows")


for name in ["diagnostic_v0", "longmemeval_smoke"]:
    summarize(name)
