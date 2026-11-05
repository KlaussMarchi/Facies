import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import glob, os, re, shutil


# ARQUIVOS (SÍSMICA, RÓTULO) DO VOLUME DE CADA DATASET EM Dataset/<dataset>/files/, LIDOS PELO Format E PELO 3 - Predict
VOLUMES = {
    'marlim': ('marlim_norm_perc_1-99_cut.npy', 'npy_multiclass_cut.npy'),
    'f3': ('f3_seismic.npy', 'f3_masks.npy')
}

# REGIÃO DO DADO REAL ((início, fim) POR EIXO) QUE O 3 - Predict MEDE; SEM ENTRADA É O VOLUME INTEIRO
VALID = {}

COLORS = ['black', '#e6194B', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4']


# ARQUIVOS DE UMA PASTA EM CAMINHO ABSOLUTO E ORDEM ALFABÉTICA
def getFiles(path, limit=None, shuffle=False):
    target = sorted(os.path.abspath(file) for file in glob.glob(os.path.join(path, '*')))

    if shuffle:
        np.random.shuffle(target)

    return target[:limit]


def setFolder(path):
    if os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs(path)


# AS TRÊS SEÇÕES CENTRAIS DE UM VOLUME, COM A MÁSCARA POR CIMA (A CLASSE 0 TRANSPARENTE) OU SOZINHA
def showTile(img=None, mask=None, save=None):
    center = [n // 2 for n in (img if img is not None else mask).shape]
    planes = lambda vol: [np.array(vol[center[0], :, :]), np.rot90(np.array(vol[:, :, center[2]]), -1), np.array(vol[:, center[1], :])]
    titles = [f'Slice X={center[0]}', f'Slice Y={center[1]}', f'Slice Z={center[2]}']

    if img is not None:
        imgPlanes = planes(img)

    if mask is not None:
        n          = max(int(np.nanmax(mask)) + 1, 2)
        colors     = COLORS[:n] if img is None else [(0, 0, 0, 0)] + COLORS[1:n]
        options    = {'cmap': ListedColormap(colors), 'vmin': 0, 'vmax': n - 1, 'alpha': None if img is None else 0.6}
        maskPlanes = planes(mask)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for i, ax in enumerate(axes):
        if img is not None:
            ax.imshow(imgPlanes[i], cmap='gray')

        if mask is not None:
            ax.imshow(maskPlanes[i], **options)

        ax.set_title(titles[i])

    plt.tight_layout()

    if save:
        plt.savefig(save, bbox_inches='tight', dpi=300)
        return plt.close(fig)

    plt.show()


# ESCALONAMENTO DOS TILES NO Format E DOS BLOCOS DO VOLUME NO 3 - Predict, PELO scaling DO info.json (COLUNA scaling DO DataBase.csv)
class Normalization:
    DEFAULT = 'percentile'    # O marlim_norm_perc_1-99_cut.npy JÁ ESTÁ NO p01/p99 DO BLOCO INTEIRO: REFAZER NO TILE MUDARIA O DADO
    options = {
        'percentile': 'o volume como vem, já no p01/p99 do bloco inteiro em [0, 1]',
        'normalize': 'min-max de cada tile para [0, 1]',
        'standardize': 'padronização de cada tile (média 0, desvio 1)',
        None: 'sem nenhum tratamento, o dado como vem'
    }

    def __init__(self, scaling=DEFAULT):
        if scaling not in self.options:
            raise ValueError(f'scaling={scaling!r} desconhecido, use um de {tuple(self.options)}')

        self.scaling = scaling

    def __call__(self, img):
        if self.scaling == 'normalize':
            return (img - np.min(img)) / (np.max(img) - np.min(img))

        if self.scaling == 'standardize':
            return (img - np.mean(img)) / np.std(img)

        return img

    def __repr__(self):
        return f'Normalization({self.scaling!r}): {self.options[self.scaling]}'

    # SEM A COLUNA É O DEFAULT, CÉLULA VAZIA É None
    @classmethod
    def read(cls, database):
        df = pd.read_csv(database, nrows=1)

        if 'scaling' not in df:
            return cls.DEFAULT

        value = df['scaling'].iloc[0]
        return value if isinstance(value, str) else None


# CORTA UM BLOCO DE target_size[0] INLINES A CADA step EM TILES target_size, COM A BORDA REFLETIDA (A MÁSCARA PELA BORDA)
class TilesBuilder:
    def __init__(self, target_size, step, normalize=None, main_dir='tiles'):
        self.size      = tuple(target_size)
        self.step      = step
        self.normalize = normalize or Normalization()
        self.img_dir   = os.path.join(main_dir, 'images')
        self.mask_dir  = os.path.join(main_dir, 'masks')

    def pad(self, vol, mode):
        pads = [(0, -n % s) for n, s in zip(vol.shape, self.size)]
        return np.pad(vol, pads, mode=mode) if any(p for _, p in pads) else vol

    def build(self, seismic, masks):
        setFolder(self.img_dir)
        setFolder(self.mask_dir)
        tz, ty, tx = self.size
        rows       = []

        for z in tqdm(range(0, seismic.shape[0], tz * self.step)):
            img = self.pad(np.asarray(seismic[z:z + tz], dtype=np.float32), 'reflect')
            msk = self.pad(np.asarray(masks[z:z + tz]).astype(np.uint8), 'edge')

            for y in range(0, img.shape[1], ty):
                for x in range(0, img.shape[2], tx):
                    tile     = self.normalize(img[:, y:y + ty, x:x + tx])
                    mask     = msk[:, y:y + ty, x:x + tx]
                    name     = f'img_{len(rows):05d}.npy'
                    imgPath  = os.path.abspath(os.path.join(self.img_dir, name))
                    maskPath = os.path.abspath(os.path.join(self.mask_dir, name))

                    np.save(imgPath, tile)
                    np.save(maskPath, mask)
                    rows.append({'img_path': imgPath, 'mask_path': maskPath, 'shape': tile.shape, 'min': float(tile.min()), 'max': float(tile.max()), 'mean': float(tile.mean()), 'std': float(tile.std()), 'msk_max': int(mask.max()), 'step': self.step, 'scaling': self.normalize.scaling})

        return rows


# REMONTA O VOLUME DOS PATCHES patch_dDDDD_hHHHH_wWWWW.dat SOBREPOSTOS, CONFERINDO SOBREPOSIÇÃO (diff, mismatch) E BURACOS (missing)
class PatchAssembler:
    PATTERN = re.compile(r'patch_d(\d+)_h(\d+)_w(\d+)\.dat$')

    def __init__(self, seismic_dir, mask_dir, size=(4, 64, 64), stride=(2, 32, 32)):
        matches          = [self.PATTERN.search(name) for name in sorted(os.listdir(seismic_dir))]
        self.seismic_dir = seismic_dir
        self.mask_dir    = mask_dir
        self.size        = tuple(size)
        self.stride      = tuple(stride)
        self.index       = {tuple(int(v) for v in match.groups()): match.string for match in matches if match}
        self.counts      = tuple(max(key[axis] for key in self.index) + 1 for axis in range(3))
        self.starts      = [self.getStarts(axis) for axis in range(3)]
        self.shape       = tuple(starts[-1] + size for starts, size in zip(self.starts, self.size))

    # O PATCH i COMEÇA EM i * stride, MENOS O ÚLTIMO, QUE ENCOSTA NO FIM E COMEÇA ONDE REPETE O FIM DO PENÚLTIMO
    def getStarts(self, axis):
        n, size, stride = self.counts[axis], self.size[axis], self.stride[axis]
        starts          = [i * stride for i in range(n)]

        if n < 2:
            return starts

        a      = self.load(self.seismic_dir, tuple(n - 2 if j == axis else 0 for j in range(3)))
        b      = self.load(self.seismic_dir, tuple(n - 1 if j == axis else 0 for j in range(3)))
        shifts = [s for s in range(1, size) if np.array_equal(np.take(a, range(s, size), axis=axis), np.take(b, range(size - s), axis=axis))]

        if not shifts:
            raise ValueError(f'eixo {axis}: o último patch não repete o fim do penúltimo em nenhum deslocamento')

        starts[-1] = starts[-2] + (stride if stride in shifts else shifts[0])
        return starts

    def build(self):
        seismic = np.zeros(self.shape, dtype=np.float32)
        masks   = np.zeros(self.shape, dtype=np.uint8)
        filled  = np.zeros(self.shape, dtype=bool)
        self.diff, self.mismatch = 0.0, 0

        for key in tqdm(self.index):
            box  = tuple(slice(self.starts[axis][key[axis]], self.starts[axis][key[axis]] + self.size[axis]) for axis in range(3))
            img  = self.load(self.seismic_dir, key)
            msk  = np.rint(self.load(self.mask_dir, key)).astype(np.uint8)
            seen = filled[box]

            self.diff      = max(self.diff, float(np.abs(seismic[box][seen] - img[seen]).max(initial=0)))
            self.mismatch += int(np.count_nonzero(masks[box][seen] != msk[seen]))
            seismic[box], masks[box], filled[box] = img, msk, True

        self.missing = int(np.count_nonzero(~filled))
        return seismic, masks

    def __repr__(self):
        return f'PatchAssembler: {len(self.index)} patches {self.size} em {self.counts}, passo {self.stride}, volume {self.shape}'

    def load(self, folder, key):
        return np.fromfile(os.path.join(folder, self.index[key]), dtype=np.float32).reshape(self.size)
