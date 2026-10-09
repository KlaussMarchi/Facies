import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import glob, os, shutil


# ARQUIVOS DE UMA PASTA (OU PADRÃO GLOB), EM CAMINHO ABSOLUTO E ORDEM ALFABÉTICA
def getFiles(path, limit=None, shuffle=False):
    target = sorted([os.path.abspath(f) for f in glob.glob(os.path.join(path, '*'))])
    if shuffle:
        np.random.shuffle(target)
    return target[:limit]


# APAGA E RECRIA A PASTA
def setFolder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


# AS TRÊS SEÇÕES CENTRAIS DE UM VOLUME, COM A MÁSCARA POR CIMA QUANDO ELA VEM: UMA COR POR FÁCIES, A CLASSE 0 TRANSPARENTE NA SOBREPOSIÇÃO
def showTile(img=None, mask=None, save=None):
    ref = img if img is not None else mask
    mx, my, mz = ref.shape[0] // 2, ref.shape[1] // 2, ref.shape[2] // 2

    def get_slices(vol):
        if vol is None:
            return None
        return [np.array(vol[mx, :, :]), np.rot90(np.array(vol[:, :, mz]), -1), np.array(vol[:, my, :])]

    img_slices, mask_slices = get_slices(img), get_slices(mask)

    if mask is not None:
        n = int(np.nanmax(mask)) + 1
        colors = ['black', '#e6194B', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4'][:max(n, 2)]
        cmap_solid   = ListedColormap(colors)
        over = list(colors); over[0] = (0, 0, 0, 0)
        cmap_overlay = ListedColormap(over)
        vmax = max(n, 2) - 1

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = [f'Slice X={mx}', f'Slice Y={my}', f'Slice Z={mz}']
    for i, ax in enumerate(axes):
        if img is not None:
            ax.imshow(img_slices[i], cmap='gray')
        if mask is not None:
            if img is not None:
                ax.imshow(mask_slices[i], cmap=cmap_overlay, vmin=0, vmax=vmax, alpha=0.6)
            else:
                ax.imshow(mask_slices[i], cmap=cmap_solid, vmin=0, vmax=vmax)
        ax.set_title(titles[i])
    plt.tight_layout()
    if save:
        plt.savefig(save, bbox_inches='tight', dpi=300); return plt.close(fig)
    plt.show()


# ESCALONAMENTO DOS TILES NO FORMAT (E DOS BLOCOS DO VOLUME NO 3 - Predict), ESCOLHIDO PELO scaling DO info.json E GRAVADO NA COLUNA scaling DO DataBase.csv
class Normalization:
    OPTIONS = ('percentile', 'normalize', 'standardize')
    DEFAULT = 'percentile'                 # O VOLUME COMO VEM: O marlim_norm_perc_1-99_cut.npy JÁ ESTÁ NO p01/p99 DO BLOCO INTEIRO, EM [0, 1]
    UNIT    = ('normalize', 'percentile')  # OS QUE DEIXAM O TILE EM [0, 1]

    def __init__(self, scaling=DEFAULT):
        if scaling not in self.OPTIONS:
            raise ValueError(f'scaling={scaling!r} desconhecido, use um de {self.OPTIONS}')

        self.scaling = scaling

    # O PERCENTIL DO CONJUNTO JÁ VEM APLICADO NO ARQUIVO: O TILE PASSA COMO ESTÁ. Refazer o p01/p99 sobre o volume já cortado
    # mudaria o dado, e não há volume cru para o null das falhas
    def __call__(self, img):
        if self.scaling == 'normalize':
            return (img - np.min(img)) / (np.max(img) - np.min(img))

        if self.scaling == 'standardize':
            return (img - np.mean(img)) / np.std(img)

        return img

    def __repr__(self):
        names = {'percentile': 'o volume como vem, já no p01/p99 do bloco inteiro em [0, 1]', 'normalize': 'min-max de cada tile para [0, 1]', 'standardize': 'padronização de cada tile (média 0, desvio 1)'}
        return f'Normalization({self.scaling!r}): {names[self.scaling]}'

    # ESCALONAMENTO DE UM DataBase.csv: SEM A COLUNA É O DEFAULT, CÉLULA VAZIA É None
    @classmethod
    def read(cls, database):
        df = pd.read_csv(database, nrows=1)

        if 'scaling' not in df:
            return cls.DEFAULT

        value = df['scaling'].iloc[0]
        return value if isinstance(value, str) else None


# CORTA UM BLOCO DE target_size[0] INLINES A CADA step DO VOLUME EM TILES target_size SEM SOBREPOSIÇÃO, COMPLETANDO A BORDA
# (REFLEXÃO NA IMAGEM, BORDA NA MÁSCARA) E ESCALONANDO CADA TILE PELA Normalization; build DEVOLVE AS LINHAS DO DataBase.csv
class TilesBuilder:
    def __init__(self, target_size, step, normalize=None, main_dir='tiles'):
        self.tz, self.ty, self.tx = target_size
        self.step      = step
        self.normalize = normalize or Normalization()
        setFolder(os.path.join(main_dir, 'images'))
        setFolder(os.path.join(main_dir, 'masks'))
        self.img_dir  = os.path.join(main_dir, 'images')
        self.mask_dir = os.path.join(main_dir, 'masks')

    def pad(self, vol, is_mask=False):
        pads = [(0, (s - vol.shape[i] % s) % s) for i, s in enumerate((self.tz, self.ty, self.tx))]
        if all(p == 0 for _, p in pads):
            return vol
        return np.pad(vol, pads, mode='edge' if is_mask else 'reflect')

    def build(self, seismic, masks):
        rows, k = [], 0
        for z in tqdm(range(0, seismic.shape[0], self.tz*self.step)):
            img = self.pad(np.asarray(seismic[z:z + self.tz], dtype=np.float32))
            msk = self.pad(np.asarray(masks[z:z + self.tz]).astype(np.uint8), is_mask=True)

            for y in range(0, img.shape[1], self.ty):
                for x in range(0, img.shape[2], self.tx):
                    it = self.normalize(img[:, y:y + self.ty, x:x + self.tx])
                    mt = msk[:, y:y + self.ty, x:x + self.tx]

                    name = f'img_{k:05d}.npy'
                    ip = os.path.abspath(os.path.join(self.img_dir,  name))
                    mp = os.path.abspath(os.path.join(self.mask_dir, name))

                    data = {
                        'img_path': ip, 'mask_path': mp, 'shape': it.shape,
                        'min': float(it.min()), 'max': float(it.max()),
                        'mean': float(it.mean()), 'std': float(it.std()),
                        'msk_max': int(mt.max()), 'step': self.step, 'scaling': self.normalize.scaling
                    }

                    np.save(ip, it)
                    np.save(mp, mt)
                    rows.append(data)
                    k += 1
        return rows
