"""Download pinned public files with validated HTTP ranges and SHA256."""
from __future__ import annotations
import argparse, hashlib, json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def fetch(url: str, dest: Path, size: int, expected_sha256: str | None = None):
    if dest.exists() and dest.stat().st_size == size:
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        if expected_sha256 is None or digest == expected_sha256:
            return digest
    chunk = 1024 * 1024
    def part(start):
        end = min(size - 1, start + chunk - 1)
        for attempt in range(4):
            try:
                separator = '&' if '?' in url else '?'
                req = urllib.request.Request(url + f'{separator}part={start}', headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(req,timeout=45) as resp:
                    content_range=resp.headers.get('Content-Range')
                    data=resp.read()
                    if resp.status != 206 or content_range != f'bytes {start}-{end}/{size}' or len(data)!=end-start+1:
                        raise ValueError(f'Bad range: {resp.status}, {content_range}, bytes={len(data)}')
                return data
            except Exception:
                if attempt==3: raise
                time.sleep(attempt+1)
    dest.parent.mkdir(parents=True,exist_ok=True)
    temporary=dest.with_suffix(dest.suffix+'.part')
    h=hashlib.sha256()
    with ThreadPoolExecutor(max_workers=4) as pool, temporary.open('wb') as out:
        for block in pool.map(part,range(0,size,chunk)):
            out.write(block);h.update(block)
    digest=h.hexdigest()
    if temporary.stat().st_size!=size or (expected_sha256 and digest!=expected_sha256):
        raise ValueError('File integrity check failed')
    temporary.replace(dest)
    return digest

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--include-s',action='store_true');a=parser.parse_args()
    root=Path(__file__).resolve().parents[1];folder=root/'data/raw/longmemeval'
    metadata=json.loads((folder/'hf_metadata.json').read_text());revision=metadata['sha']
    url=f'https://huggingface.co/api/datasets/xiaowu0162/longmemeval-cleaned/tree/{revision}'
    tree=json.load(urllib.request.urlopen(url))
    selected=['longmemeval_oracle.json']+(['longmemeval_s_cleaned.json'] if a.include_s else [])
    records=[]
    for item in tree:
        if item['path'] not in selected: continue
        name=item['path'];source=f'https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/{revision}/{name}'
        print('Downloading',name,item['size'],flush=True)
        sha=fetch(source,folder/name,item['size'],item.get('lfs',{}).get('oid'))
        data=json.loads((folder/name).read_text(encoding='utf8'))
        records.append({'file':name,'url':source,'revision':revision,'bytes':item['size'],'sha256':sha,'records':len(data)})
        print('Verified',name,len(data),'records',flush=True)
    (root/'data/longmemeval_download_manifest.json').write_text(json.dumps(records,indent=2),encoding='utf8')
