import sys,time,json,copy
from pathlib import Path
import numpy as np
r=Path('.').resolve();sys.path.insert(0,str(r/'src'))
from memory_rerank.embeddings import DenseEncoder
import torch
enc=DenseEncoder(r)
x=json.loads((r/'data/raw/longmemeval/longmemeval_s_cleaned.json').read_text(encoding='utf8'))[0]
texts=[]
for session in x['haystack_sessions']:
 for t in session:
  tokens=enc.model.tokenizer.encode(t['content'],add_special_tokens=False)[:200]
  if len(tokens)>=180:texts.append(enc.model.tokenizer.decode(tokens,skip_special_tokens=True))
  if len(texts)==16:break
 if len(texts)==16:break
out=[]; reference=None
for threads,batch in [(4,2),(8,8),(12,8),(4,8)]:
 torch.set_num_threads(threads)
 start=time.perf_counter();v=enc.model.encode(texts,batch_size=batch,normalize_embeddings=True,show_progress_bar=False)
 elapsed=time.perf_counter()-start
 if reference is None:reference=v
 row={'precision':'fp32','threads':threads,'batch':batch,'seconds':elapsed,'min_cosine_vs_reference':float(np.min(np.sum(v*reference,axis=1)))}
 out.append(row);print(row,flush=True)
start=time.perf_counter()
enc.model=torch.ao.quantization.quantize_dynamic(enc.model,{torch.nn.Linear},dtype=torch.qint8)
print('Quantized in',time.perf_counter()-start,flush=True)
for threads,batch in [(4,8),(8,8)]:
 torch.set_num_threads(threads)
 start=time.perf_counter();v=enc.model.encode(texts,batch_size=batch,normalize_embeddings=True,show_progress_bar=False)
 row={'precision':'int8_dynamic','threads':threads,'batch':batch,'seconds':time.perf_counter()-start,'min_cosine_vs_reference':float(np.min(np.sum(v*reference,axis=1)))}
 out.append(row);print(row,flush=True)
(r/'reports/cpu_encoding_probe.json').write_text(json.dumps({'scope':'16 long excerpts throughput probe only, not retrieval quality','results':out},indent=2),encoding='utf8')
