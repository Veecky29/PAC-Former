import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np

def read(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))

def write(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        output = csv.DictWriter(stream, fieldnames=list(rows[0]))
        output.writeheader()
        output.writerows(rows)

def summarize(source, output):
    rows = read(source)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['dataset'], int(row['horizon']), row['variant']].append(row)
    summary = []
    for (dataset, horizon, variant), group in grouped.items():
        group.sort(key=lambda r: int(r['seed']))
        if [int(r['seed']) for r in group] != [2026, 2027, 2028]:
            raise ValueError('Every setting requires seeds 2026, 2027 and 2028')
        item = dict(dataset=dataset, horizon=horizon, variant=variant)
        for metric in ['mse', 'mae']:
            values = np.array([float(r['test_' + metric]) for r in group])
            item[metric + '_mean'] = float(values.mean())
            item[metric + '_sd'] = float(values.std(ddof=1))
        summary.append(item)
    write(output, summary)
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, default=Path('results/runs.csv'))
    parser.add_argument('--out', type=Path, default=Path('results/ablation_summary.csv'))
    args = parser.parse_args()
    summarize(args.runs, args.out)
if __name__ == '__main__':
    main()
