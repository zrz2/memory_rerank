"""Run/resume at most 12 budgeted calls on fixed BGE candidate pools."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from memory_rerank.llm import DeepSeekClient
from memory_rerank.llm_scoring import BudgetedScorer,save_json
from memory_rerank.scoring import rank
from memory_rerank.metrics import evaluate

def main():
    cfg=json.loads((ROOT/'configs/pilot_v1.json').read_text(encoding='utf8'))
    output=ROOT/'runs/pilot_v1'
    output.mkdir(parents=True,exist_ok=True)
    lock=output/'api.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.close(fd)
    try:
        scorer=BudgetedScorer(DeepSeekClient(ROOT),output/'api',cfg)
        ids=[x['question_id'] for x in json.loads((ROOT/'data/splits/longmemeval_smoke_ids.json').read_text(encoding='utf8'))]
        for qid in ids:
            prepared=output/f'{qid}_prepared.json'
            target=output/f'{qid}_result.json'
            if target.exists():
                print(qid,'result exists',flush=True)
                continue
            if not prepared.exists():
                print(qid,'not prepared yet',flush=True)
                continue
            p=json.loads(prepared.read_text(encoding='utf8'))
            if p['protocol']!=cfg: raise ValueError('Preparation protocol differs')
            start=time.perf_counter()
            importance,ir,ic=scorer.call('importance',p['views'])
            reranked,rr,rc=scorer.call('rerank',p['views'],p['query'])
            by_id={p['alias_to_id'][alias]:score for alias,score in importance.items()}
            memories=[{**m,'importance':by_id[m['id']],'importance_source':'deepseek_prefix_batch_v1'} for m in p['candidates']]
            methods={}
            for label,method in [('semantic_top20','semantic'),('semantic_recency_top20','semantic_recency'),('ga_importance_top20','ga_paper_score')]:
                methods[label]=rank(memories,p['similarities'],p['query'],method,cfg['decay'])
            methods['deepseek_rerank_top20']=[{'memory_id':p['alias_to_id'][alias],'position':i+1} for i,alias in enumerate(reranked)]
            methods['semantic_full']=p['full_semantic']
            methods['semantic_recency_full']=p['full_semantic_recency']
            gold={'relevance_grades':{mid:1 for mid in p['gold_ids']}}
            rows=[{'method':method,'metric':evaluate([x['memory_id'] for x in ranked],gold,k)} for method,ranked in methods.items() for k in cfg['ks']]
            result={'question_id':qid,'question_type':p['question_type'],'candidate_recall':evaluate([m['id'] for m in memories],gold,cfg['candidate_n'])['recall'],
                    'gold_count':len(p['gold_ids']),'candidate_count':len(memories),'truncated_candidate_count':len(p['truncated_candidate_ids']),
                    'importance_scores':by_id,'rankings':methods,'rows':rows,'api_keys':[ir['key'],rr['key']],
                    'response_models':[ir['response'].get('returned_model'),rr['response'].get('returned_model')],
                    'usage':[ir['response'].get('usage'),rr['response'].get('usage')], 'cache_hits':[ic,rc],
                    'elapsed_seconds':time.perf_counter()-start,'preparation_fingerprint':p['fingerprint'],
                    'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    'git_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())}
            save_json(target,result)
            print(qid,'DONE',result['candidate_recall'],result['usage'],flush=True)
        print('Budget reserved:',sum(x['reserved_cny'] for x in scorer.ledger['calls']),flush=True)
    finally: lock.unlink()
if __name__=='__main__': main()
