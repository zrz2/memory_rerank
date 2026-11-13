"""Run the first local diagnostic; it is a pipeline smoke test, not a claim of performance."""
from __future__ import annotations
import hashlib, json, platform, subprocess, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from memory_rerank.embeddings import DenseEncoder
from memory_rerank.metrics import evaluate
from memory_rerank.scoring import rank

ROOT=Path(__file__).resolve().parents[1]
tasks=json.loads((ROOT/'data/diagnostic/tasks_v0.json').read_text(encoding='utf8'))
encoder=DenseEncoder(ROOT); rows=[]; start=time.perf_counter()
for task in tasks:
    contents=[m['content'] for m in task['memories']]
    emb=encoder.encode(contents,'passage')
    for q in task['queries']:
        qe=encoder.encode([q['text']],'query')[0]
        sims={m['id']:float(emb[i]@qe) for i,m in enumerate(task['memories'])}
        for method in ['semantic','semantic_recency','ga_paper_score','ga_task_oracle','ga_task_validity_oracle']:
            ranking=rank(task['memories'],sims,q,method)
            for k in [1,3,5]:
                metric=evaluate([x['memory_id'] for x in ranking],q,k)
                rows.append({'task_id':task['task_id'],'query_id':q['query_id'],'method':method,'ranking':[x['memory_id'] for x in ranking],'metric':metric,'scores':ranking})
out=ROOT/'runs/diagnostic_v0.json';out.write_text(json.dumps({'status':'smoke','created_on':'2026-11-13','tasks':len(tasks),'query_count':sum(len(t['queries']) for t in tasks),'python':platform.python_version(),'model':encoder.config,'data_sha256':hashlib.sha256((ROOT/'data/diagnostic/tasks_v0.json').read_bytes()).hexdigest(),'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'git_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),'elapsed_seconds':time.perf_counter()-start,'encoder_stats':encoder.stats,'rows':rows},ensure_ascii=False,indent=2),encoding='utf8')
print('wrote',out,'rows',len(rows),'stats',encoder.stats)
