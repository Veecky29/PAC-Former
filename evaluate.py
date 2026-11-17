import argparse
import json
from pathlib import Path
import numpy as np
import torch
from pacformer import training as t
from pacformer.model import create_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+', choices=t.DATASETS, default=t.DATASETS)
    parser.add_argument('--horizons', nargs='+', type=int, choices=t.HORIZONS, default=t.HORIZONS)
    parser.add_argument('--data-dir', type=Path, default=t.ROOT / 'data' / 'benchmark')
    parser.add_argument('--out', type=Path, default=Path('outputs/evaluation'))
    args = parser.parse_args()
    t.DATA_DIR = args.data_dir
    t.setup_cuda()
    audit = json.loads((t.ROOT / 'configs/datasets.json').read_text())
    rows = []
    for dataset in args.datasets:
        series = t.Series(dataset, audit, args.out)
        try:
            for horizon in args.horizons:
                payload = torch.load(args.models / 'checkpoints' / dataset / f'H{horizon}.pt', map_location='cpu', weights_only=True)
                config = payload['selection']['config']
                model = create_model(series.channels, horizon, config)
                model.load_state_dict(payload['model_state'], strict=True)
                result = t.evaluate(model, series, horizon, 'test')
                reference = payload['metrics']['test']
                np.testing.assert_allclose([result['mse'], result['mae']], [reference['mse'], reference['mae']], rtol=2e-06, atol=2e-07)
                rows.append(dict(dataset=dataset, horizon=horizon, **result))
                del model
        finally:
            series.close()
    args.out.mkdir(parents=True, exist_ok=True)
    t.write_csv(args.out / 'metrics.csv', rows)
    print(json.dumps(rows, indent=2))
if __name__ == '__main__':
    main()
