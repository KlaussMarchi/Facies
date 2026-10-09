# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Segmentação de fácies sísmicas 3D (6 classes) no bloco real de Marlim: corta o volume interpretado em tiles, treina redes
3D neles e prediz o volume inteiro nos inlines que ficaram de fora. É o irmão do repositório `Falhas`: mesma estrutura e
mesmo código, que só muda onde a segmentação multiclasse e o tile `(4, 256, 256)` exigem (lista no fim). Não há build,
lint nem suíte de testes — a verificação é rodar o notebook de cima para baixo e comparar os números.

## Ambiente e execução

Tudo roda no python do `base` do conda (torch 2.7.1+cu128, monai 1.5.2, torchmetrics, papermill, opencv, scipy); o kernel
`python3` dos notebooks resolve para ele. O env `fault_pred` **não tem** torchmetrics nem papermill.

- **Campanha de treinos (o caminho normal):** edite `Task/task.json` (lista de rodadas, cada uma no formato de
  `Task/info.json`) e rode `cd Task && python index.py`. Para cada rodada ele grava `Task/info.json`, executa
  `Dataset/<dataset>/Format.ipynb` (pulado quando o `DataBase.csv` já existe cortado com o `img_size`, o `step` e o
  `normalize` da rodada; `img_size`/`step` null aceitam o que existir) e depois, uma vez por trial (`n_trials`; o trial
  vai no `info.json` e a semente é `42 + trial`), `Model/1 - Model.ipynb` e, se ele terminou, `Model/3 - Predict.ipynb`
  no modelo recém-salvo. Tudo via papermill, com a saída em `Task/logs/<nome>_out.ipynb`.
- **Um notebook só:** abra no Jupyter/VS Code com o kernel do `base`, ou
  `papermill <nb> /tmp/out.ipynb -k python3 --cwd <pasta do nb>` — todo caminho dentro de um notebook é relativo à
  **pasta dele**, então o `cwd` errado quebra tudo.
- GPU: AMP só quando o `info.json` tem `"amp": true` (padrão desligado; a campanha do Facies liga em toda rodada), `epochs`
  (padrão 100; as rodadas usam 200) e early stopping sobre `val_iou` com `patience` (padrão 15; as rodadas usam 20). O
  `1 - Model` liga `expandable_segments` no alocador da GPU. Uma rodada custa horas; nunca re-execute um treino para
  "verificar" sem pedir.
- `.gitignore` exclui `*.npy`, `*.pth`, `*.dat` e `*.zip`: o volume (`files/`, ~17 GB), os tiles e os pesos são locais; as
  figuras dos backups entram no git.

## Fluxo dos dados

Os estágios trocam **arquivos**, nunca variáveis, e cada um lê o que o anterior gravou:

1. **`Dataset/facies/files/`** — `marlim_norm_perc_1-99_cut.npy`, a sísmica float32 `(3137, 552, 2050)` =
   `(inline, z, xline)` já no p01/p99 do bloco em `[0, 1]`, e `npy_multiclass_cut.npy`, o rótulo uint8 com as classes
   0–5. A classe 0 é uma fácies como as outras, **não é fundo**.
2. **`Dataset/facies/Format.ipynb`.** Corta um bloco de `img_size[0]` inlines a cada `step` (padrão 3) em tiles
   `img_size`, com a borda do volume preenchida por reflexão (o rótulo pela borda), e grava `tiles/{images,masks}` e o
   `DataBase.csv` (`img_path`/`mask_path` absolutos, `shape`, estatísticas, `msk_max`, `step`, `normalize`). Com
   `"normalize": true` (padrão) o tile é o volume como vem; com `false` é a padronização de cada tile (média 0, desvio 1,
   o passo do artigo da ResACEUnet). Sem a coluna `normalize`, é o `true`; o `1 - Model`, o `Model_CV` e o `3 - Predict`
   param com erro se ela não bater com a rodada. Também reescreve `Task/info.json` com o nome do dataset.
3. **`Task/info.json`** é a configuração única da rodada (`network`, `dataset`, `img_size`, `step`, `lr`, `loss`,
   `batch_size`, `scheduler`, `dropout`, `num_filters`, `ema`, `n_trials`, opcionais `epochs`, `patience`, `amp`,
   `normalize`, `deep_supervision` e `augmentations`; `multiclass` é só informativo), lida como `OPTIONS`.
4. **`Model/1 - Model.ipynb`** — o treino. Split fixo (`random_state=42`) de 80/10/10 dos tiles, `classes` =
   `msk_max` + 1, `CustomDataset`, `Trainer` (clip de gradiente, `plateau` = `ReduceLROnPlateau` por época, `cosine` =
   aquecimento 1e-6 → `lr` em 10 épocas e cosseno até 1e-7 a cada batch (a agenda da ResACEUnet), `ModelEMA` do
   `Model/EMA/` quando `ema` é true, deep supervision com pesos 1/0.5/0.25 quando a rede devolve lista, progresso
   corrente em `Model/progress.json`). Salva `Model/Backup/model_N/` com `info.json`, `model.pth`
   (`{'model', 'optimizer', 'timestamp', 'history'}`), `train.png`, `classes.png` e `predictions/`; `N` é o maior
   `model_N` + 1. No `info.json` salvo, `processing` é o `Task/info.json` exatamente como veio (copiar e colar reproduz a
   rodada), `division` guarda o split (`val_size`, `test_size`, `step`, `n_images`, e `k_fold` no CV), `model` os
   argumentos da rede (o `model.img_size` vem do `shape` do `DataBase.csv` ou, com `crop`, da janela do recorte) e `iou`
   o IoU de cada classe no teste. **`Model/0 - Model_CV.ipynb`** é o mesmo treino em K dobras (salva a melhor).
5. **`Model/2 - Compare.ipynb`** junta todo `info.json` da pasta `BACKUP` numa tabela (o `marlim_iou` entra como a
   média) e compara variações com a `VariationAnalysis` (média±std entre trials).
6. **`Model/3 - Predict.ipynb`** — pega o último modelo de `BASE_PATH` (ou o `modelId` fixado), refaz o split de teste
   pelo `division` (confere `n_images`, `normalize` e `step`) e mede pela matriz de confusão `classes × classes` dos
   voxels (precisão, recall, IoU e F1 por classe; a média do IoU é o `test_iou`). Depois prediz o **volume inteiro** em
   blocos do tile, pulando os inlines que viraram tile, com a mesma normalização, o mesmo `amp` e a mesma predição do
   treino (`Transforms.infer`), e grava `marlim_iou` no `info.json`, `marlim_classes.png` e `views/`.
7. **`Documents/Backups/Backup1|Backup2/model_N/`** guardam campanhas antigas no formato de `Model/Backup` (de antes do
   `division`: o `3 - Predict` lê `n_images`/`step` do `processing` e usa o `test_size` 0.1 de então); os notebooks
   trocam a pasta pelo `BACKUP`/`BASE_PATH`.

## Convenções que atravessam o repositório

- **Módulo é pasta com `index.py`** exportando o tipo principal: `from Network.index import ModelNetwork`,
  `from Losses.index import Losses`, `from Transforms.index import Transforms`. Notebook fora da pasta ajusta o path
  (`sys.path.append('..')`). Importar módulo nunca executa trabalho.
- **Eixos:** o tile é `(inline, z, xline)`, o mesmo `(x, z, y)` do Falhas, mas o inline tem só 4 amostras. Toda rede
  deriva o pool/passo de cada eixo do `img_size` (`getPools`/`_compute_pools`: um eixo só é reduzido enquanto tem
  tamanho, nos níveis mais fundos); no cubo isso é idêntico ao Falhas.
- **Nova rede:** arquivo em `Model/Network/types/X.py`, import e um `if` em `ModelNetwork.get()` com uma linha
  MAIÚSCULA citando a origem; o nome usado ali é o que vai em `Task/info.json` e é o mesmo do Falhas. Hoje: `unet_3d`,
  `dbrnet`, `segresnet` (com pad até múltiplo de 8), `resaceunet_grva` (a única com `deep_supervision`),
  `resaceunet_wu`, `resaceunet_zu` (passos por eixo; no cubo, bit a bit as do Falhas), `macnn`, `fault_seg_net`,
  `nru_net`, `fault_edge_former` (`num_filters` múltiplo de 9). Nome desconhecido levanta erro; os antigos `standard`,
  `unet3d_v2` e `resaceunet` viraram `unet_3d`, `dbrnet` e `resaceunet_grva` (os backups foram migrados).
- **Nova loss:** classe em `Model/Losses/index.py` e entrada em `Losses.options` (`cross_entropy`, `dice_focal`,
  `dice_ce`, `focal`, `smooth_dice`, `compound`, `tversky`). Toda loss força `float32` fora do autocast. No multiclasse o
  `alpha` do `focal` não é aplicado (o MONAI daria 1 − alpha à classe 0 como se fosse fundo). No MONAI 1.5.2 o termo
  focal do `DiceFocalLoss` é sempre sigmoide, mesmo no caso multiclasse.
- **Aumentação:** só pelo `augmentations` do `info.json` (`null`/ausente = tiles do `Format` como estão).
  `Model/Transforms/index.py` é o do Falhas (receita da ResACEUnet + `zoom`, sorteio no `DataLoader` com semente
  `(42 + trial, época, índice)`, `n_aug`, na ordem do JSON e só no treino; com `crop` a rede é montada na janela e
  valida/testa/prediz o tile inteiro por `Transforms.infer`) com o rótulo multiclasse: a máscara segue a imagem pelo
  vizinho mais próximo, a borda da rotação e da suavização espelha o tile (borda zero seria sísmica morta com a classe
  0), o `crop` centra numa classe sorteada por igual entre as do tile e o `rotate90` só aceita plano quadrado. No tile
  `(inline, z, xline)`: `flip` em `[0, 2]`, `rotate` em `[[1, 2]]`, recorte do tipo `[4, 128, 128]`; nada que vire o `z`.
- **Idioma:** identificadores em inglês e `camelCase`; comentários, markdown, títulos de gráfico, commits e relatórios
  em português.
- **Estilo de escrita:** uma linha de comentário MAIÚSCULA acima de cada classe, sem docstring, sem type hint, sem
  underscore inicial, chamada nunca quebrada em várias linhas; nos notebooks, uma etapa por célula terminando numa prova
  visível (print, tabela ou figura), e a explicação vai no markdown da seção, em tópicos.

## O que difere do Falhas (de propósito)

Ao levar uma mudança de um repositório para o outro, só isto deve continuar diferente:

- Multiclasse: `MULTICLASS = True` com `classes` do `msk_max`, `getClasses` (IoU por classe) e a paleta `FACIES_COLORS`
  no `utils`/`Plotter`/`plotPrediction`; o `alpha` do `focal` desligado no multiclasse; o `Transforms` com rótulo por
  vizinho mais próximo, borda espelhada e `crop` por classe; o `3 - Predict` com matriz `classes × classes` e sem o
  protocolo de recortes da Tabela 2 da ResACEUnet (que é de falha).
- Tile `(4, 256, 256)`: pools/passos por eixo nas redes, `Segresnet` com pad, split 80/10/10, `step` no `division`, o
  `Format` que corta o volume, o `Task/index.py` que confere `img_size`/`step` e roda o `3 - Predict` depois do treino,
  e o lote do teste do `3 - Predict` igual ao `batch_size` do treino (a conta de 128³ voxels por lote das falhas daria 8
  tiles e estoura a GPU de 12 GB na `dbrnet`).
- Sem `Synthetic/`, `Marlim/` e `Nature/`: a predição no bloco real é o volume inteiro do `3 - Predict`.
