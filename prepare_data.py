import argparse
import hashlib
import json
import shutil
import time
import urllib.request
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'configs/datasets.json').read_text())['datasets']
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', choices=list(manifest), default=list(manifest))
    parser.add_argument('--out', type=Path, default=root / 'data/benchmark')
    parser.add_argument('--source-dir', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        spec = manifest[name]
        target = args.out / (name + '.csv')
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == spec['sha256']:
            print(name, 'ready')
            continue
        temporary = target.with_suffix('.download')
        if args.source_dir:
            shutil.copyfile(args.source_dir / (name + '.csv'), temporary)
        else:
            for attempt in range(3):
                try:
                    request = urllib.request.Request(spec['source_url'], headers={'User-Agent': 'Python/3.11'})
                    with urllib.request.urlopen(request, timeout=120) as source, temporary.open('wb') as output:
                        shutil.copyfileobj(source, output)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2 * (attempt + 1))
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != spec['sha256']:
            raise ValueError(f'{name}: file checksum does not match the dataset manifest')
        temporary.replace(target)
        print(name, 'ready')
if __name__ == '__main__':
    main()
