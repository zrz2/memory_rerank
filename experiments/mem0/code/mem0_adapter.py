"""Mem0 transport adapter: keep upstream prompts, use project DeepSeek and a persisted budget."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from mem0.llms.base import LLMBase
from memory_rerank.llm import DeepSeekClient
from memory_rerank.llm_scoring import save_json

class AuditedDeepSeek(LLMBase):
    def __init__(self,config=None):
        super().__init__(config)
        self.client=DeepSeekClient(ROOT)
        self.folder=ROOT/'experiments/mem0/runs/api/smoke';self.folder.mkdir(parents=True,exist_ok=True)
        self.path=self.folder/'ledger.json'
        self.ledger=json.loads(self.path.read_text(encoding='utf8')) if self.path.exists() else {'calls':[]}
    def generate_response(self,messages,response_format=None,tools=None,tool_choice=None,**kwargs):
        if tools or tool_choice:raise ValueError('Tool calls not part of this bounded smoke test')
        limit=1600
        identity={'messages':messages,'response_format':response_format,'llm':self.client.config,'max_tokens':limit}
        key=hashlib.sha256(json.dumps(identity,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        cache=self.folder/f'{key}.json'
        if cache.exists():
            response=json.loads(cache.read_text(encoding='utf8'))['response']
        else:
            if any(x['key']==key for x in self.ledger['calls']):raise RuntimeError('Unresolved previous request; no paid retry')
            other=ROOT/'runs/pilot_v1/api/ledger.json'
            prior=sum(x['reserved_cny'] for x in json.loads(other.read_text())['calls']) if other.exists() else 0
            reserve=((len(json.dumps(messages,ensure_ascii=False).encode())+2048)*10+limit*30)/1e6
            if prior+sum(x['reserved_cny'] for x in self.ledger['calls'])+reserve>5 or len(self.ledger['calls'])>=8:
                raise RuntimeError('Shared 5 CNY budget or 8-call Mem0 limit exceeded')
            entry={'key':key,'reserved_cny':reserve,'status':'reserved'}
            self.ledger['calls'].append(entry);save_json(self.path,self.ledger)
            response=self.client.chat(messages,max_tokens=limit,response_format=response_format)
            save_json(cache,{**identity,'response':response})
            entry.update(status='response_saved',usage=response.get('usage'),returned_model=response.get('returned_model'))
            save_json(self.path,self.ledger)
        if response.get('finish_reason')!='stop':raise ValueError('Incomplete Mem0 LLM output')
        parsed=json.loads(response['text'])
        if not isinstance(parsed,dict) or not isinstance(parsed.get('memory'),list):raise ValueError('Unexpected extraction schema')
        return response['text']
