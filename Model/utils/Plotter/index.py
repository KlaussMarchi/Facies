import matplotlib.pyplot as plt
import numpy as np
try:
    from utils.index import FACIES_COLORS
except ImportError:
    from Model.utils.index import FACIES_COLORS


# BARRA DE CLASSE ('0', '1', ...) TEM A COR DA FACIES NO showTile E A 'mean' FICA CINZA; TRES CASAS PORQUE OS
# IOUS DAS FACIES SE SEPARAM NA TERCEIRA
class Plotter:
    def __init__(self, metrics, limits=(0, 1.10), title='Metric Analysis', stds=None):
        keys   = [str(key) for key in metrics.keys()]
        values = list(metrics.values())
        
        if isinstance(stds, dict):
            std_values = [stds.get(k, np.nan) for k in metrics.keys()]
        elif isinstance(stds, list):
            std_values = stds
        else:
            std_values = [np.nan] * len(values)
            
        n = len(metrics)
        base   = plt.cm.tab10(np.arange(n)) if n <= 10 else plt.cm.viridis(np.linspace(0, 1, n))
        colors = [FACIES_COLORS[int(key)] if (key.isdigit() and int(key) < len(FACIES_COLORS)) else ('#708090' if key.lower() == 'mean' else base[i]) for i, key in enumerate(keys)]
        bars   = plt.bar(keys, values, color=colors)

        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.title(title)

        if limits:
            plt.ylim(limits)

        for bar, std in zip(bars, std_values):
            height = bar.get_height()
            offset = (plt.ylim()[1] - plt.ylim()[0]) * 0.02
            xData  = (bar.get_x() + bar.get_width() / 2.)
            label_text = f'{height:.3f} ± {std:.3f}' if (std == std and std is not None) else f'{height:.3f}'
            plt.text(xData, (height + offset), label_text, ha='center', va='bottom', rotation=35 if len(keys) > 4 else 0, fontsize=11, fontweight='bold', color='black')

        if len(keys) > 4:
            plt.xticks(rotation=45, ha='right')
