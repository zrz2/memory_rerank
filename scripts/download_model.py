"""Fetch only files required by the pinned dense embedding model."""
import hashlib,json,urllib.request
from pathlib import Path
from download_longmemeval import fetch
r=Path(__file__).resolve().parents[1];config=json.loads((r/'configs/embedding_model.json').read_text())
model=config['model'];rev=config['revision'];folder=r/'.cache/models/multilingual-e5-small'
tree=json.load(urllib.request.urlopen(f'https://huggingface.co/api/models/{model}/tree/{rev}?recursive=true'))
names={'1_Pooling/config.json','config.json','model.safetensors','modules.json','sentence_bert_config.json','sentencepiece.bpe.model','special_tokens_map.json','tokenizer.json','tokenizer_config.json','README.md'}
manifest=[]
for item in tree:
 name=item['path']
 if name not in names:continue
 url=f'https://huggingface.co/{model}/resolve/{rev}/{name}';target=folder/name;target.parent.mkdir(parents=True,exist_ok=True)
 print('Fetching',name,item['size'],flush=True)
 if item['size']>1024*1024:
  sha=fetch(url,target,item['size'],item.get('lfs',{}).get('oid'))
 else:
  data=urllib.request.urlopen(url,timeout=40).read()
  if len(data)!=item['size']:raise ValueError('length mismatch')
  target.write_bytes(data);sha=hashlib.sha256(data).hexdigest()
 manifest.append({'file':name,'sha256':sha,'bytes':item['size']})
(r/'configs/embedding_files_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
print('Model files verified.',flush=True)
