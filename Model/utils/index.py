import glob, os, shutil, cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import ListedColormap


def getFiles(path, limit=None, shuffle=False):
    target = sorted(glob.glob(os.path.join(path, '*')))
    if shuffle:
        np.random.shuffle(target) 
    return target[:limit]

def getAllFiles(base):
    return [os.path.join(root, file) for root, dirs, files in os.walk(base) for file in files]

def getFile(path, index):
    return getFiles(path)[index]

def discretize(img, thresh=127):
    return cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]

def setFolder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

# ── Seleção de GPUs ───────────────────────────────────────────────────────────
# double=True usa todas as GPUs (o batch é dividido entre elas); False mantém só a principal.

def canShareMemory(ids, size=1 << 20):
    """A cópia direta entre as GPUs é confiável? Com IOMMU/VT-d ativo ela volta
    corrompida sem erro nenhum, e o DataParallel treinaria com lixo em metade do batch."""
    ref = torch.randn(size)
    src = ref.to(f'cuda:{ids[0]}')
    return all(torch.equal(src.to(f'cuda:{i}').cpu(), ref) for i in ids[1:])


def getDevices(double=False, verbose=True):
    """Retorna os ids das GPUs usadas no treino (lista vazia = CPU)."""
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')  # menos fragmentação = mais VRAM útil

    total  = torch.cuda.device_count() if torch.cuda.is_available() else 0
    ids    = list(range(total)) if double else list(range(min(total, 1)))
    broken = (len(ids) > 1) and not canShareMemory(ids)

    if broken:
        ids = ids[:1]
        print('! multi-GPU desligado: a cópia entre as placas volta corrompida (IOMMU/VT-d bloqueando o DMA)')
        print('  corrigir com intel_iommu=off no GRUB e reiniciar; até lá o treino roda em uma GPU só')

    if not verbose:
        return ids

    memory = lambda i: torch.cuda.get_device_properties(i).total_memory / 1024**3
    device = f'cuda:{ids[0]}' if ids else 'cpu'

    print('── GPU SETUP ' + '─' * 38)
    print(f'  torch      : {torch.__version__} (cuda {torch.version.cuda})')
    print(f'  alocador   : {os.environ["PYTORCH_CUDA_ALLOC_CONF"]}')
    print(f'  detectadas : {total}')

    for i in range(total):
        print(f'    [{i}] {torch.cuda.get_device_properties(i).name} — {memory(i):.0f} GB  {"(em uso)" if i in ids else "(ignorada)"}')

    if len(ids) > 1:
        print(f'  modo       : DataParallel em {len(ids)} GPUs — {sum(memory(i) for i in ids):.0f} GB somados')
    else:
        print(f'  modo       : {"GPU única" if ids else "CPU"}{" (P2P quebrado)" if broken else ""}')

    print(f'  principal  : {device}')
    print('─' * 51)
    return ids


# ── Paleta discreta multiclasse (facies) ──────────────────────────────────────
# Índice da lista == id da classe. Classe 0 = fundo/primeira facies (preto).
FACIES_COLORS = [
    '#000000',  # 0
    '#e6194B',  # 1  vermelho
    '#3cb44b',  # 2  verde
    '#4363d8',  # 3  azul
    '#f58231',  # 4  laranja
    '#911eb4',  # 5  roxo
    '#42d4f4',  # 6  ciano
    '#f032e6',  # 7  magenta
    '#bfef45',  # 8  lima
    '#fabed4',  # 9  rosa
]


def faciesCmaps(n_classes):
    """Retorna (cmap_solido, cmap_overlay) com `n_classes` cores discretas.
    A classe 0 é totalmente preenchida na cor preta (sem transparência de fundo)."""
    n_classes = max(int(n_classes), 2)
    if n_classes <= len(FACIES_COLORS):
        base = list(FACIES_COLORS[:n_classes])
    else:
        base = [plt.cm.tab20(i % 20) for i in range(n_classes)]

    solid = ListedColormap(base)
    overlay_cols = list(base)
    return solid, ListedColormap(overlay_cols)


def showTile(img=None, mask=None, save=None, classes=None):
    """Mostra os 3 planos médios de um tile 3D com legenda associando o número de cada classe à sua cor."""
    if img is None and mask is None:
        return print("Erro: Forneça pelo menos 'img' ou 'mask'.")

    ref_vol = img if img is not None else mask
    mid_x = ref_vol.shape[0] // 2
    mid_y = ref_vol.shape[1] // 2
    mid_z = ref_vol.shape[2] // 2

    def get_slices(vol):
        if vol is None:
            return None

        s_x = np.array(vol[mid_x, :, :]) # Plano YZ
        s_y = np.array(vol[:, mid_y, :]) # Plano XZ
        s_z = np.array(vol[:, :, mid_z]) # Plano XY
        return [s_x, np.rot90(s_z, -1), s_y]

    img_slices  = get_slices(img)
    mask_slices = get_slices(mask)

    if mask is not None:
        n_classes = classes if classes is not None else int(np.nanmax(mask)) + 1
        cmap_mask_only, cmap_mask_overlay = faciesCmaps(n_classes)
        vmax = max(int(n_classes), 2) - 1

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    titles    = [f'Slice X={mid_x}', f'Slice Y={mid_y}', f'Slice Z={mid_z}']

    for i, ax in enumerate(axes):
        if img is not None:
            ax.imshow(img_slices[i], cmap='gray')

        if mask is not None:
            if img is not None:
                ax.imshow(mask_slices[i], cmap=cmap_mask_overlay, vmin=0, vmax=vmax, alpha=0.6)
            else:
                ax.imshow(mask_slices[i], cmap=cmap_mask_only, vmin=0, vmax=vmax)

        ax.set_title(titles[i], fontsize=12)
        ax.axis('off')

    if mask is not None:
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=cmap_mask_only.colors[c], edgecolor='white', label=f'Classe {c}') for c in range(n_classes)]
        fig.legend(handles=legend_elements, loc='lower center', ncol=min(n_classes, 7), bbox_to_anchor=(0.5, 0.01), fontsize=11)
        plt.tight_layout(rect=[0, 0.08, 1, 1])
    else:
        plt.tight_layout()

    if save:
        plt.savefig(save, bbox_inches='tight', dpi=200)
        return plt.close(fig)

    plt.show()