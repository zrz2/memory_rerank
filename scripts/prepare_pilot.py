"""Prepare BGE candidates without exposing labels to the model; resumable per query."""
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from memory_rerank.embeddings import DenseEncoder
from memory_rerank.scoring import available, rank
from run_longmemeval_smoke import iso

def main():
    cfg = json.loads((ROOT/'configs/pilot_v1.json').read_text(encoding='utf8'))
    model = json.loads((ROOT/'configs/embedding_model.json').read_text(encoding='utf8'))
    source = ROOT/'data/raw/longmemeval/longmemeval_s_cleaned.json'
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    selected = json.loads((ROOT/'data/splits/longmemeval_smoke_ids.json').read_text(encoding='utf8'))
    ids = {x['question_id'] for x in selected}
    items = [x for x in json.loads(source.read_text(encoding='utf8')) if x['question_id'] in ids]
    output = ROOT/'runs/pilot_v1'
    output.mkdir(parents=True, exist_ok=True)
    encoder = None
    for item in items:
        qid = item['question_id']
        target = output/f'{qid}_prepared.json'
        fingerprint = hashlib.sha256(json.dumps([cfg,model,source_hash],sort_keys=True).encode()).hexdigest()
        if target.exists():
            saved = json.loads(target.read_text(encoding='utf8'))
            if saved['fingerprint'] != fingerprint: raise ValueError('Existing preparation config mismatch')
            print(qid, 'checkpoint exists', flush=True)
            continue
        if encoder is None: encoder = DenseEncoder(ROOT)
        start = time.perf_counter()
        query = {'text':item['question'], 'query_time':iso(item['question_date'])}
        memories, grades = [], {}
        for sid, date, turns in zip(item['haystack_session_ids'],item['haystack_dates'],item['haystack_sessions']):
            for i,t in enumerate(turns):
                mid=f'{sid}:{i}'
                memories.append({'id':mid,'content':f"{t['role']}: {t['content']}", 'created_at':iso(date),'last_accessed_at':iso(date),'importance':5})
                if t.get('has_answer'): grades[mid]=1
        visible=available(memories,query['query_time'])
        if set(grades)-{m['id'] for m in visible}: raise ValueError('Future gold evidence')
        chunks, owners=[],[]
        for m in visible:
            tokens=encoder.model.tokenizer.encode(m['content'],add_special_tokens=False)
            for offset in range(0,max(1,len(tokens)),cfg['chunk_stride']):
                chunks.append(encoder.model.tokenizer.decode(tokens[offset:offset+cfg['chunk_tokens']],skip_special_tokens=True))
                owners.append(m['id'])
                if offset+cfg['chunk_tokens']>=len(tokens): break
        print(qid, len(visible),'turns',len(chunks),'chunks',flush=True)
        before=dict(encoder.stats)
        # Bound each inference call and show progress; vectors remain cached per text.
        vector_blocks=[]
        for offset in range(0,len(chunks),64):
            vector_blocks.append(encoder.encode(chunks[offset:offset+64],'passage'))
            print(qid, 'encoded',min(offset+64,len(chunks)),'/',len(chunks),flush=True)
        vectors=np.concatenate(vector_blocks)
        qvec=encoder.encode([query['text']],'query')[0]
        sims={m['id']:-1.0 for m in visible}
        for mid,score in zip(owners,vectors@qvec): sims[mid]=max(sims[mid],float(score))
        full_semantic=rank(visible,sims,query,'semantic')
        full_recency=rank(visible,sims,query,'semantic_recency',cfg['decay'])
        candidate_ids=[x['memory_id'] for x in full_semantic[:cfg['candidate_n']]]
        lookup={m['id']:m for m in visible}
        candidates=[lookup[mid] for mid in candidate_ids]
        shuffled=list(candidates)
        random.Random(str(cfg['seed'])+qid).shuffle(shuffled)
        views=[{'alias':f'M{i+1:03d}','created_at':m['created_at'],'text':m['content'][:cfg['memory_view_max_chars']]} for i,m in enumerate(shuffled)]
        payload={'fingerprint':fingerprint,'question_id':qid,'question_type':item['question_type'],'query':query,
                 'gold_ids':sorted(grades),'candidates':candidates,'similarities':{mid:sims[mid] for mid in candidate_ids},
                 'alias_to_id':{v['alias']:m['id'] for v,m in zip(views,shuffled)},'views':views,
                 'truncated_candidate_ids':[m['id'] for m in candidates if len(m['content'])>cfg['memory_view_max_chars']],
                 'full_semantic':full_semantic,'full_semantic_recency':full_recency,'visible_turns':len(visible),
                 'chunks':len(chunks),'source_sha256':source_hash,'embedding_config':model,'protocol':cfg,
                 'encoder_stats_delta':{k:encoder.stats[k]-before[k] for k in before},'elapsed_seconds':time.perf_counter()-start,
                 'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                 'git_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())}
        target.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8')
        print(qid,'PREPARED',round(payload['elapsed_seconds'],1),'seconds',flush=True)
if __name__=='__main__': main()
