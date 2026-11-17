import argparse
import hashlib
import json
import logging
from pathlib import Path
from pacformer import training as t

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['search', 'main', 'ablation'], required=True)
    parser.add_argument('--datasets', nargs='+', choices=t.DATASETS, default=t.DATASETS)
    parser.add_argument('--horizons', nargs='+', type=int, choices=t.HORIZONS, default=t.HORIZONS)
    parser.add_argument('--data-dir', type=Path, default=t.ROOT / 'data' / 'benchmark')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reuse-main', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t.DATA_DIR = args.data_dir
    t.setup_cuda()
    audit = json.loads((t.ROOT / 'configs' / 'datasets.json').read_text())
    selected = json.loads((t.ROOT / 'configs' / 'selected.json').read_text())
    protocol = json.loads((t.ROOT / 'configs' / 'training.json').read_text())
    protocol['mode'] = args.mode
    protocol['model_sha256'] = t.sha(t.ROOT / 'pacformer' / 'model.py')
    protocol['training_sha256'] = t.sha(Path(t.__file__))
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if (args.out / 'protocol.json').exists():
        if json.loads((args.out / 'protocol.json').read_text()) != protocol:
            raise ValueError('Output directory contains a different configuration')
    else:
        t.atomic_json(args.out / 'protocol.json', protocol)
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    for dataset in args.datasets:
        series = t.Series(dataset, audit, args.out)
        try:
            for horizon in args.horizons:
                group = args.out / dataset / f'H{horizon}'
                config = selected[dataset][str(horizon)]['config']
                if args.mode == 'search':
                    trials = []
                    t.ACTIVE_VARIANT = 'F'
                    for index, candidate in enumerate(t.GRID, 1):
                        result = t.fit_job(series, horizon, candidate, 2026, group / 'search' / f'trial_{index:02d}', 'grid_search', protocol_hash, logger, lambda _: None)
                        trials.append(result)
                    winner = min(range(len(trials)), key=lambda i: (trials[i]['best_val_mse'], i))
                    config = trials[winner]['config']
                    t.atomic_json(group / 'selected.json', dict(trial=winner + 1, config=config, val_mse=trials[winner]['best_val_mse']))
                variants = ['B', 'P', 'S', 'F'] if args.mode == 'ablation' else ['F']
                seeds = [2026, 2027, 2028] if args.mode == 'ablation' else [2026]
                for variant in variants:
                    t.ACTIVE_VARIANT = variant
                    for seed in seeds:
                        path = group / variant / f'seed_{seed}'
                        if args.mode == 'ablation' and variant == 'F' and (seed == 2026) and args.reuse_main:
                            source = args.reuse_main / 'checkpoints' / dataset / f'H{horizon}.pt'
                            t.reuse_main_checkpoint(source, series, horizon, config, path, protocol_hash)
                            continue
                        t.fit_job(series, horizon, config, seed, path, 'final_refit', protocol_hash, logger, lambda _: None)
        finally:
            series.close()
if __name__ == '__main__':
    main()
