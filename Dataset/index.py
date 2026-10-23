import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import glob, os, re, shutil


# ARQUIVOS DO VOLUME (SÍSMICA, RÓTULO) DE CADA DATASET EM Dataset/<dataset>/files/, LIDOS PELO Format E PELO 3 - Predict
VOLUMES = {
    'marlim': ('marlim_norm_perc_1-99_cut.npy', 'npy_multiclass_cut.npy'),
    'f3':     ('f3_seismic.npy', 'f3_masks.npy'),
}

# REGIÃO DO DADO REAL DE CADA VOLUME ((início, fim) POR EIXO), A ÚNICA QUE O 3 - Predict MEDE; SEM ENTRADA É O VOLUME INTEIRO.
# OS PATCHES DO F3 COBREM O VOLUME (601, 255, 901) DO BENCHMARK DE ALAUDAH ET AL. (2019) COM BORDA REFLETIDA DE 1 INLINE,
# 16 AMOSTRAS E 16 XLINES DE CADA LADO: A BORDA REPETE VOXELS DE DENTRO E NÃO ENTRA NA COMPARAÇÃO COM OS ESTUDOS
VALID = {
    'f3': ((1, 602), (16, 271), (16, 917)),
}


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
    OPTIONS = ('percentile', 'normalize', 'standardize', None)
    DEFAULT = 'percentile'                 # O VOLUME COMO VEM: O marlim_norm_perc_1-99_cut.npy JÁ ESTÁ NO p01/p99 DO BLOCO INTEIRO, EM [0, 1]
    UNIT    = ('normalize', 'percentile')  # OS QUE DEIXAM O TILE EM [0, 1]

    def __init__(self, scaling=DEFAULT):
        if scaling not in self.OPTIONS:
            raise ValueError(f'scaling={scaling!r} desconhecido, use um de {self.OPTIONS}')

        self.scaling = scaling

    # O PERCENTIL DO CONJUNTO JÁ VEM APLICADO NO ARQUIVO: O TILE PASSA COMO ESTÁ. Refazer o p01/p99 sobre o volume já cortado
    # mudaria o dado. None (null no info.json) É SEM NENHUM TRATAMENTO: O DADO JÁ VEM PRONTO, COMO OS PATCHES DO F3
    def __call__(self, img):
        if self.scaling == 'normalize':
            return (img - np.min(img)) / (np.max(img) - np.min(img))

        if self.scaling == 'standardize':
            return (img - np.mean(img)) / np.std(img)

        return img

    def __repr__(self):
        names = {'percentile': 'o volume como vem, já no p01/p99 do bloco inteiro em [0, 1]', 'normalize': 'min-max de cada tile para [0, 1]', 'standardize': 'padronização de cada tile (média 0, desvio 1)', None: 'sem nenhum tratamento, o dado como vem'}
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


# REMONTA O VOLUME DOS PATCHES BRUTOS patch_dDDDD_hHHHH_wWWWW.dat (float32 NO FORMATO size, SÍSMICA E RÓTULO COM O MESMO NOME):
# O PATCH i DE UM EIXO COMEÇA EM i * stride, MENOS O ÚLTIMO, QUE ENCOSTA NO FIM DO VOLUME E TEM O INÍCIO ACHADO PELA SOBREPOSIÇÃO
# COM O PENÚLTIMO; build CONFERE QUE TODA SOBREPOSIÇÃO REPETE O MESMO DADO (diff, mismatch) E QUE NENHUM VOXEL FICOU VAZIO (missing)
class PatchAssembler:
    PATTERN = re.compile(r'patch_d(\d+)_h(\d+)_w(\d+)\.dat$')

    def __init__(self, seismic_dir, mask_dir, size=(4, 64, 64), stride=(2, 32, 32)):
        self.seismic_dir = seismic_dir
        self.mask_dir    = mask_dir
        self.size        = tuple(size)
        self.stride      = tuple(stride)
        self.index  = {tuple(int(v) for v in self.PATTERN.search(name).groups()): name for name in sorted(os.listdir(seismic_dir)) if self.PATTERN.search(name)}
        self.counts = tuple(max(key[axis] for key in self.index) + 1 for axis in range(3))
        self.starts = [self.getStarts(axis) for axis in range(3)]
        self.shape  = tuple(self.starts[axis][-1] + self.size[axis] for axis in range(3))

    def load(self, folder, key):
        return np.fromfile(os.path.join(folder, self.index[key]), dtype=np.float32).reshape(self.size)

    # INÍCIOS DE UM EIXO: A GRADE DO stride E O ÚLTIMO NO DESLOCAMENTO EM QUE ELE REPETE O FIM DO PENÚLTIMO
    def getStarts(self, axis):
        n, size, stride = self.counts[axis], self.size[axis], self.stride[axis]
        starts = [i * stride for i in range(n)]
        if n < 2:
            return starts

        a = self.load(self.seismic_dir, tuple(n - 2 if j == axis else 0 for j in range(3)))
        b = self.load(self.seismic_dir, tuple(n - 1 if j == axis else 0 for j in range(3)))
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

        for key, name in tqdm(self.index.items()):
            s   = tuple(slice(self.starts[axis][key[axis]], self.starts[axis][key[axis]] + self.size[axis]) for axis in range(3))
            img = self.load(self.seismic_dir, key)
            msk = np.rint(self.load(self.mask_dir, key)).astype(np.uint8)

            seen = filled[s]
            self.diff      = max(self.diff, float(np.abs(seismic[s][seen] - img[seen]).max(initial=0)))
            self.mismatch += int(np.count_nonzero(masks[s][seen] != msk[seen]))
            seismic[s], masks[s], filled[s] = img, msk, True

        self.missing = int(np.count_nonzero(~filled))
        return seismic, masks

    def __repr__(self):
        return f'PatchAssembler: {len(self.index)} patches {self.size} em {self.counts}, passo {self.stride}, volume {self.shape}'
