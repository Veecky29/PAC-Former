import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=Path('results/ablation_summary.csv'))
    parser.add_argument('--out', type=Path, default=Path('figures'))
    args = parser.parse_args()
    with args.results.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    args.out.mkdir(parents=True, exist_ok=True)
    datasets = list(dict.fromkeys((r['dataset'] for r in rows)))
    plt.rcParams.update({'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})
    colors = {'B': '#58616f', 'P': '#007ca8', 'S': '#d68b00', 'F': '#009c7a'}
    for metric in ['mse', 'mae']:
        fig, axes = plt.subplots(2, 4, figsize=(13.4, 6.8), layout='constrained')
        for ax, dataset in zip(axes.flat, datasets):
            for variant, color in colors.items():
                values = sorted([r for r in rows if r['dataset'] == dataset and r['variant'] == variant], key=lambda r: int(r['horizon']))
                ax.errorbar(range(4), [float(r[metric + '_mean']) for r in values], yerr=[float(r[metric + '_sd']) for r in values], marker='o', capsize=2, color=color, label=variant)
            ax.set(title=dataset, xticks=range(4), xticklabels=[96, 192, 336, 720], xlabel='Forecast horizon', ylabel=metric.upper())
            ax.grid(alpha=0.2)
        axes.flat[-1].axis('off')
        handles, labels = axes.flat[0].get_legend_handles_labels()
        axes.flat[-1].legend(handles, ['Backbone', 'PLM only', 'SCA only', 'Complete model'], loc='center', frameon=False)
        for extension in ['png', 'svg']:
            fig.savefig(args.out / f'ablation_{metric}.{extension}', dpi=200, bbox_inches='tight')
        plt.close(fig)
if __name__ == '__main__':
    main()
