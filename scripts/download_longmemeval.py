"""Download pinned public files with validated HTTP ranges and SHA256."""
from __future__ import annotations
import argparse, hashlib, json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def fetch(url: str, dest: Path, size: int, expected_sha256: str | None = None):
    if dest.exists() and dest.stat().st_size == size:
        with dest.open('rb') as existing:
            digest = hashlib.file_digest(existing, 'sha256').hexdigest()
        if expected_sha256 is None or digest == expected_sha256:
            return digest
    chunk = 1024 * 1024
    def part(start):
        end = min(size - 1, start + chunk - 1)
        for attempt in range(6):
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
                if attempt==5: raise
                time.sleep(attempt+1)
    dest.parent.mkdir(parents=True,exist_ok=True)
    temporary=dest.with_suffix(dest.suffix+'.part')
    # Resume previously validated complete ranges; verify final full-file hash.
    resume = temporary.stat().st_size if temporary.exists() else 0
    if resume > size:
        raise ValueError('Partial file larger than source')
    h=hashlib.sha256()
    if resume:
        with temporary.open('rb') as existing:
            while block := existing.read(chunk): h.update(block)
    with ThreadPoolExecutor(max_workers=16) as pool, temporary.open('ab') as out:
        # Bound buffered responses to 16 MiB even when an earlier range retries.
        for batch_start in range(resume, size, 16 * chunk):
            starts = range(batch_start, min(size, batch_start + 16 * chunk), chunk)
            for block in pool.map(part, starts):
                out.write(block);h.update(block)
            out.flush()
            print(f'Downloaded {out.tell()}/{size} bytes', flush=True)
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
