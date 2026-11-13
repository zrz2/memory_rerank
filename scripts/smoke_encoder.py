from pathlib import Path
import sys,json
sys.path.insert(0,'E:/毕业论文/memory-rerank/src')
from memory_rerank.embeddings import DenseEncoder
enc=DenseEncoder(Path('E:/毕业论文/memory-rerank'))
a=enc.encode(['我的收货地址改为上海。','我喜欢爵士乐。'],'passage')
q=enc.encode(['今天快递寄到哪里？'],'query')
print(a.shape,q.shape,enc.stats,float(a[0]@q[0]),float(a[1]@q[0]))
