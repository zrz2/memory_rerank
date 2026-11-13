from pathlib import Path
import sys,json
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
from memory_rerank.embeddings import DenseEncoder
enc=DenseEncoder(root)
a=enc.encode(['我的收货地址改为上海。','我喜欢爵士乐。'],'passage')
q=enc.encode(['今天快递寄到哪里？'],'query')
print(a.shape,q.shape,enc.stats,float(a[0]@q[0]),float(a[1]@q[0]))
