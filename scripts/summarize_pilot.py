"""Summarize complete six-question pilot, retaining every question and failure."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
ROOT=Path(__file__).resolve().parents[1]
PILOT=ROOT/'runs/pilot_v1'
LABELS={'semantic_top20':'BGE-M3 纯语义','semantic_recency_top20':'语义+近因性（同候选）','ga_importance_top20':'GA式三因子（同候选）','deepseek_rerank_top20':'DeepSeek 重排（同候选）','semantic_full':'纯语义（完整历史）','semantic_recency_full':'语义+近因性（完整历史）'}

def main():
    ids=[x['question_id'] for x in json.loads((ROOT/'data/splits/longmemeval_smoke_ids.json').read_text(encoding='utf8'))]
    results=[json.loads((PILOT/f'{qid}_result.json').read_text(encoding='utf8')) for qid in ids]
    prepared={qid:json.loads((PILOT/f'{qid}_prepared.json').read_text(encoding='utf8')) for qid in ids}
    groups=defaultdict(list)
    for result in results:
        for row in result['rows']:groups[(row['method'],row['metric']['k'])].append(row['metric'])
    aggregate=[]
    for (method,k),values in sorted(groups.items()):
        aggregate.append({'method':method,'k':k,**{metric:mean(x[metric] for x in values) for metric in ['recall','mrr','ndcg','complete_evidence']}})
    ledger=json.loads((PILOT/'api/ledger.json').read_text(encoding='utf8'))
    calls=ledger['calls']
    prompt_tokens=sum(x.get('usage',{}).get('prompt_tokens',0) for x in calls)
    completion_tokens=sum(x.get('usage',{}).get('completion_tokens',0) for x in calls)
    models=sorted({x.get('returned_model','unknown') for x in calls})
    estimated=(prompt_tokens*2+completion_tokens*8)/1e6 if models==['deepseek-flash'] else None
    summary={'scope':'six_question_pilot_not_official_benchmark','aggregate':aggregate,'question_ids':ids,
             'candidate_recall20_macro':mean(x['candidate_recall'] for x in results),
             'api_calls':len(calls),'prompt_tokens':prompt_tokens,'completion_tokens':completion_tokens,
             'reserved_cny':sum(x['reserved_cny'] for x in calls),'flash_peak_cost_estimate_cny':estimated,'returned_models':models,
             'truncated_candidate_views':sum(x['truncated_candidate_count'] for x in results),
             'input_sha256':{qid:hashlib.sha256((PILOT/f'{qid}_result.json').read_bytes()).hexdigest() for qid in ids},
             'per_question':[{k:r[k] for k in ['question_id','question_type','candidate_recall','gold_count','truncated_candidate_count','rows']} for r in results]}
    report=ROOT/'reports/pilot_v1_summary.json'
    report.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    lines=['# 下一阶段结果：相同候选上的记忆重排序','',
           '**本报告是六题流程试验，不是正式论文结论。所有题均保留，没有按结果选题或调参。**','',
           '## 这次究竟做了什么','',
           '从每道题的完整历史中，用 BGE-M3 找出前20条记忆，再比较四种排序。DeepSeek 给出真实 importance 分数，并独立做一次基于问题的重排。排序之后才读取正确证据标签来计算指标。','',
           '打分看的是长期重要程度；重排看的是对当前问题的证据价值。这两个概念不能混为一谈。模型看不到答案、证据标记和原始数据ID。','',
           '## 主要结果：前5条记忆','',
           '|方法（同一个Top20候选池）|Recall@5|MRR@5|NDCG@5|全部已标证据覆盖率|',
           '|---|---:|---:|---:|---:|']
    for row in aggregate:
        if row['k']==5 and row['method'].endswith('top20'):
            lines.append('|'+LABELS[row['method']]+'|'+'|'.join(f"{row[m]:.4f}" for m in ['recall','mrr','ndcg','complete_evidence'])+'|')
    lines += ['', 'Recall@5：已标注证据中，有多少进入前5条。MRR@5：第一条已标证据越靠前越高。NDCG@5：已标证据整体是否靠前。全部已标证据覆盖率不等于回答正确率。', '',
              f"候选 Recall@20 的逐题平均值为 **{summary['candidate_recall20_macro']:.4f}**。候选阶段没找到的证据，后面的重排无论多强都找不回来。",'',
              '## 每道题都列出，避免只看平均值','',
              '|题型|候选Recall@20|纯语义R@5|语义+近因性R@5|GA式R@5|DeepSeek R@5|','|---|---:|---:|---:|---:|---:|']
    for r in results:
        values={x['method']:x['metric']['recall'] for x in r['rows'] if x['metric']['k']==5}
        lines.append(f"|{r['question_type']}|{r['candidate_recall']:.4f}|"+'|'.join(f'{values[m]:.4f}' for m in ['semantic_top20','semantic_recency_top20','ga_importance_top20','deepseek_rerank_top20'])+'|')
    lines += ['','## 候选范围为什么必须说明','',
              '完整历史上的近因性排序与Top20内的近因性排序，不是同一个实验：候选内容和归一化范围都不同。以下单独报告，不把候选截断的影响误当成某个打分因子的贡献。','',
              '|完整历史对照|Recall@5|MRR@5|','|---|---:|---:|']
    for row in aggregate:
        if row['k']==5 and row['method'].endswith('_full'): lines.append(f"|{LABELS[row['method']]}|{row['recall']:.4f}|{row['mrr']:.4f}|")
    lines += ['','## 逐题检查线索','']
    for r in results:
        p=prepared[r['question_id']];gold=set(p['gold_ids']);byid={m['id']:m for m in p['candidates']}
        semantic=[x['memory_id'] for x in r['rankings']['semantic_top20'][:5]]
        llm=[x['memory_id'] for x in r['rankings']['deepseek_rerank_top20'][:5]]
        gained=sorted((set(llm)-set(semantic))&gold);lost=sorted((set(semantic)-set(llm))&gold)
        lines += [f"### {r['question_type']} / {r['question_id']}",'',p['query']['text'],'',
                  f"- 已标证据 {len(gold)} 条；候选中找到 {len(gold & set(byid))} 条。",
                  f"- DeepSeek相对纯语义进入前5的已标证据：{', '.join(gained) or '无'}；离开前5的已标证据：{', '.join(lost) or '无'}。",
                  f"- 20条候选中有 {r['truncated_candidate_count']} 条仅给LLM前1600字符。"]
        for mid in (gained+lost)[:2]:
            snippet=byid[mid]['content'][:260].replace('\n',' ')
            lines += ['',f'> {snippet}']
        lines += ['']
    lines += ['## 费用与复现','',
              f"- 实际请求 {len(calls)} 次，输入 {prompt_tokens} tokens，输出 {completion_tokens} tokens。",
              f"- 请求前保守预留合计 {summary['reserved_cny']:.4f} 元，上限5元。预留不是扣款。",
              f"- 按核验当日Flash高峰价、把所有输入均按未缓存估算：{estimated:.4f} 元（平台账单才是实际扣款依据）。" if estimated is not None else '- 模型身份变化，需按响应身份另核价格。',
              f"- 返回模型：{', '.join(models)}；请求名保持deepseek-chat。",
              '- 逐题完整记录：runs/pilot_v1；API原始响应与预算账本：runs/pilot_v1/api；汇总JSON：reports/pilot_v1_summary.json。',
              '- 运行：python scripts/prepare_pilot.py → python scripts/run_pilot.py → python scripts/summarize_pilot.py。已有结果与响应缓存会复用。','',
              '## 现在能说明什么，不能说明什么','',
              '- 可以核查同候选池中排序是否改变、变化是否命中已标证据，以及费用和耗时。',
              '- 六题已用于开发检查，不能再包装成独立测试集；没有统计显著性结论。',
              '- GA式方法是三项打分公式的迁移：候选范围、模型、批量importance、截断视图和时间代理均与原论文不同，不是Smallville原系统复现。',
              '- LLM只看固定前1600字符，可能看不到证据；BGE召回看完整切块。需要后续全文／窗口视图对照。',
              '- 未标has_answer的记忆可能仍有帮助；没有做官方回答生成和judge，不能把检索分数叫作问答准确率。',
              '- DeepSeek重排是LLM对照，不能替代尚未加入的开源Cross-Encoder。',
              '- 下一步应冻结独立dev/test、补强语义重排对照，再评估自动时间有效性和任务状态方法；本轮不提出“已经创新成功”的结论。','']
    (ROOT/'reports/pilot_v1_summary.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps({k:summary[k] for k in ['candidate_recall20_macro','api_calls','prompt_tokens','completion_tokens','reserved_cny','flash_peak_cost_estimate_cny']},indent=2))
if __name__=='__main__':main()
