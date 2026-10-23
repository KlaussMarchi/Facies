# Comparação com a literatura: fácies do F3 fora do `step`

Este relatório compara os modelos `model_1` (DBRNet) e `model_2` (ResACE-UNet GRVA) com os estudos publicados mais
parecidos, usando a métrica medida nas inlines que ficaram fora do `step`.

**Resumo**

- O volume `f3` do repositório é o F3 do benchmark de Alaudah et al. (2019), com 6 classes e `601 × 255 × 901`
  (inline × amostra × xline). Ele veio dos patches com uma borda refletida (1 inline, 16 amostras e 16 xlines de cada
  lado), que **agora fica fora da métrica** (`VALID` em `Dataset/index.py`).
- Dois estudos usam o **mesmo protocolo** daqui no mesmo F3: inlines rotuladas a intervalo regular e teste em todas
  as outras inlines do volume. Os dois rotulam muito menos (~1 %) e usam redes 2D:
  - **CONSS (Li et al., 2023):** 7 inlines rotuladas;
  - **SeisSegDiff (Ore & Gao, 2025):** 5 inlines rotuladas.
- Não encontrei estudo no F3 com esse protocolo e uma fração de treino perto dos 33 % daqui. Também não achei nenhum com
  rede 3D.
- Completam a comparação:
  - **Pham et al. (2026):** redes 3D por voxel no mesmo volume, com validação em blocos contíguos de xlines.
  - **Alaudah et al. (2019):** o benchmark que define os dados e as métricas.
  - **Civitarese et al. (2019):** divisão intercalada em blocos com poucas seções, mas no Penobscot.
  - **Li et al. (2026):** mede o quanto a vizinhança entre inlines de treino e de teste infla a métrica.
- Fora do `step`, os modelos dão **mIoU 90,9 % (model_1) e 90,5 % (model_2)**, contra 87,1 % do CONSS supervisionado,
  90,7 % do CONSS semissupervisionado e 80,7 % do SeisSegDiff.
- Essa vantagem **não mostra que a rede é melhor**: aqui se usam 201 inlines rotuladas (33 %) contra 5–7 (~1 %), e
  toda inline de teste fica a 1–4 inlines de uma de treino, contra até ~43–60 nos dois estudos.
- O próprio `block_iou` do Marlim confirma isso. Com a mesma rede e o mesmo treino, ele cai de 95,7 (`step` 3) para
  93,5 (`step` 5) e 87,6 (`step` 10) (seção 6).
- A seção 7 diz como fazer a comparação justa: `step` 25 e 30 reproduzem o regime do CONSS e do SeisSegDiff sem mexer
  no código.

## 1. O que está sendo comparado

### 1.1 Dados

- **F3 Netherlands** (Mar do Norte, setor holandês), com a interpretação de Alaudah et al. (2019). São 7 grupos
  litoestratigráficos interpretados por horizontes e perfis de poço; Rijnland e Chalk foram fundidos, o que dá 6 classes.
- O volume anotado completo é o `601 × 901 × 255` (inlines 100–700 × xlines 300–1200 × 255 amostras), o mesmo usado
  pelo CONSS e por Pham et al.
- O `Dataset/f3/Format.ipynb` remonta os patches `(4, 64, 64)` num volume `(603, 287, 933)`. Conferi voxel a voxel que
  a diferença é **borda refletida**:
  - `inline[0] == inline[2]` e `inline[602] == inline[600]`;
  - `z[16 − k] == z[16 + k]` e `z[270 + k] == z[270 − k]`;
  - o mesmo vale nas xlines, em torno de 16 e de 916;
  - isso vale para a sísmica e para o rótulo.
- A região real é `inlines 1..601, z 16..270, xlines 16..916`.

**Classes (correspondência entre os trabalhos e frequência no volume real)**

| Classe aqui | Grupo (Alaudah, CONSS, Li) | Pham et al. | Voxels no volume real |
|---|---|---|---|
| 0 | Upper North Sea | Class 1 | 25,3 % |
| 1 | Middle North Sea | Class 2 | 10,9 % |
| 2 | Lower North Sea | Class 3 | 49,5 % |
| 3 | Rijnland/Chalk | Class 4 | 6,3 % |
| 4 | Scruff | Class 5 | 6,2 % |
| 5 | Zechstein | Class 6 | 1,8 % |

### 1.2 Protocolo dos modelos 1 e 2

**Corte (`Format`, `step` 3)**

- Um bloco de 4 inlines a cada 12 vira tile: 51 blocos × 75 tiles = 3825 tiles `(4, 64, 64)`.
- Os tiles são sorteados em 80/10/10 (`random_state` 42): 3059 de treino, 383 de validação e 383 de teste.
- No volume real, as inlines que viraram tile são **201 de 601 (33,4 %)**, e só 80 % dos tiles delas entram no gradiente.

**Treino**

- `dbrnet` (model_1) e `resaceunet_grva` (model_2), com `dice_focal`, `plateau`, EMA, AMP, lr 1e-4, lote 2,
  32 filtros, dropout 0,05.
- Até 200 épocas, com parada antecipada (paciência 20) sobre o `val_iou`.
- Um trial cada (semente 42).

**Avaliação fora do `step` (`3 - Predict`)**

- O volume é predito em blocos de 4 × 64 × 64 e a métrica entra nas **400 inlines (66,6 %) que nunca viraram tile**:
  91.902.000 voxels = 400 × 255 × 901.
- Os blocos de teste ficam entre blocos de treino, então a distância até a inline de treino mais próxima é de 1, 2, 3 ou
  4 inlines (média 2,5).
- A seleção de modelo usa só os tiles de validação, que vêm das inlines de treino. Nada fora do `step` entra no treino
  nem na escolha do modelo.

### 1.3 Métricas

Agora todas saem da mesma matriz de confusão de voxels (linha é a classe real, coluna a predita), somada antes de
dividir. O `3 - Predict` grava em `block_metrics` no `info.json` de cada modelo:

| Chave | Definição | Quem usa |
|---|---|---|
| `pa` | acertos / total | Alaudah, CONSS, Li |
| `class_accuracy`, `mca` | `tp / real` de cada classe (o recall) e a média simples | Alaudah, CONSS, Li |
| `iou`, `miou` | `tp / (tp + fp + fn)` e a média simples | CONSS, Li (o `block_iou` continua igual) |
| `fwiu` | Σ (frequência real da classe × IoU da classe) | Alaudah, Li |
| `f1`, `mf1` | `2tp / (2tp + fp + fn)` = Dice da classe e a média simples | CONSS (F1), Pham (Dice) |
| `precision` | `tp / predito` | só leitura |

O Civitarese et al. usa o **mmIoU** (o mIoU de cada imagem, médio sobre as 40 imagens de teste), que não é a mesma
conta; ele não foi implementado.

## 2. Como a pesquisa foi feita

**Critérios, em ordem de peso**

1. Mesmo dado: o F3 de Alaudah com 6 classes.
2. Mesmo protocolo: seções rotuladas amostradas dentro do volume e métrica nas seções restantes do **mesmo** volume.
3. Rede 3D.
4. Números por classe com uma das métricas acima.

**Buscas**

- Fontes: arXiv, IEEE Xplore, Frontiers, Wiley, as APIs do Semantic Scholar, do Crossref e do Unpaywall, e o GitHub dos
  autores.
- Termos: benchmark do F3 (Alaudah); segmentação 3D de fácies no F3; inlines esparsas ("every n-th inline", "1 % labels",
  "few labeled slices"); semissupervisionado no F3; vazamento espacial em divisões aleatórias; e as citações dos achados
  (o protocolo de Canguçu et al., os trabalhos que citam o CONSS).

**Resultado da busca**

- Nenhum trabalho cumpre os quatro critérios juntos.
- O CONSS e o SeisSegDiff cumprem 1, 2 e 4, e Pham et al., 1, 3 e 4. Os outros três entram pelo protocolo
  (Civitarese), pela referência (Alaudah) e pela medida do vazamento (Li 2026).
- Procurei também, sem achar, estudos no F3 com o protocolo intercalado e 10–60 % das inlines no treino. O único
  candidato é o Abid et al. (2022), com divisão aleatória de 60 % das seções, mas com 7 classes e sem texto completo
  acessível.
- Todos os PDFs baixados são de acesso aberto (arXiv, IEEE JSTARS, Frontiers e o manuscrito aceito do SeisSegDiff no
  NSF PAR).

**Descartados ou só citados**

| Trabalho | Por que ficou de fora |
|---|---|
| Abid et al. (2022), *Wirel. Commun. Mob. Comput.*, doi:10.1155/2022/7762543 | F3 com divisão **aleatória** 60/20/20 de seções (treino e teste intercalados, como aqui), mas com 7 classes: GPA 98,52 %, MCA 96,88 %, mIoU 93,92 % (ensemble). O download automático foi bloqueado (Cloudflare). Gutierrez et al. (2025) criticam essa divisão porque seções vizinhas são quase iguais. |
| Gutierrez et al. (2025), *Geophys. Prospecting*, doi:10.1111/1365-2478.70104 (CC-BY) | Comparação justa de 5 redes 2D (DeepLab V3/V3+, Segmenter, SegFormer, SETR) no F3 e no SEAM AI; aponta a falta de detalhes de divisão nos trabalhos. O download foi bloqueado (Cloudflare); baixe pelo navegador. |
| Canguçu et al. (2025), *IEEE TGRS* 63 | O "principled benchmark" de 5 dobras espaciais no F3, Penobscot e Parihaka, que Pham et al. seguem. Acesso pago. |
| Silva et al. (2019), arXiv:1904.00770 | Outra interpretação do F3 (IBM, 10 classes) e outro rótulo. |
| Silva et al. (2025), arXiv:2512.00191 | Treina em uma inline a cada 10/20/40, mas a tarefa é horizonte, não fácies. |
| Nasim et al. (2022), arXiv:2011.10510; Saha & Whitaker (2025), arXiv:2501.16760 | Usam a divisão oficial de Alaudah (adaptação de domínio) ou aprendizado *few-shot* treinado em outros volumes: são outros paradigmas. |
| Tolstaya & Egorov (2022), *Interpretation* 10(2), doi:10.1190/INT-2021-0140.1 | Divisão em blocos contíguos ao longo das xlines (a que Gutierrez et al. seguem), não intercalada. |
| Thouvenot et al. (2025), EUSIPCO, "Geo2.5D" | 3D semissupervisionado com 3 seções de referência, mas só no Parihaka. |

## 3. Os estudos (PDFs nesta pasta)

### 3.1 CONSS: Li et al. (2023), `1_Li2023_CONSS_JSTARS.pdf`

O protocolo mais parecido com o deste repositório.

- **Referência:** K. Li, W. Liu, Y. Dou, Z. Xu, H. Duan, R. Jing. *CONSS: Contrastive Learning Method for
  Semisupervised Seismic Facies Classification*. IEEE JSTARS 16, 7838–7849, 2023. doi:10.1109/JSTARS.2023.3308754.
  Preprint em arXiv:2210.04776; código em github.com/upcliuwenlong/CONSS_SEISMIC_FACIES (sem o *loader* da divisão).
- **Dados:** o F3 de Alaudah completo, `255 × 901 × 601` (amostra × xline × inline), 6 classes.
- **Divisão:**
  - o volume é dividido uniformemente em 7 blocos ao longo das inlines;
  - a **primeira inline de cada bloco** vai para o treino (7 inlines, cerca de 1 % do volume);
  - **todas as outras inlines vão para o teste**.
  - É o mesmo desenho do `step` daqui, com o espaçamento ~86 inlines e 1 inline por posição.
- **Rede e treino:**
  - DeepLabv3+ 2D em janelas de 256 × 256 deslizando na xline (28 janelas rotuladas e 2376 sem rótulo);
  - Adam lr 1e-4, StepLR (passo 4, gamma 0,1), 8 épocas;
  - os métodos semissupervisionados usam a **sísmica das inlines de teste sem rótulo** no treino (transdutivo); o
    supervisionado não usa.
- **Métricas:** PA, acurácia por classe, MCA, mIoU e F1 médio, todas por pixel somado no conjunto de teste.
- **Resultados (Tabela I, %):**

| Método | PA | Upper | Middle | Lower | R/C | Scruff | Zechstein | MCA | mIoU | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Supervisionado | 96,62 | 98,38 | 93,27 | 99,14 | 89,41 | 92,70 | 79,53 | 92,09 | 87,12 | 92,87 |
| Pseudolabel | 97,04 | 98,42 | 93,45 | 99,11 | 90,84 | 92,85 | 79,88 | 92,43 | 87,83 | 93,31 |
| Mean teacher | 97,38 | 98,80 | 94,57 | 98,76 | 92,82 | 93,26 | 86,92 | 94,19 | 89,06 | 94,05 |
| Cross pseudo-supervision | 97,44 | 98,91 | 94,33 | 99,15 | 92,73 | 91,11 | 87,93 | 94,02 | 89,24 | 94,16 |
| Cross-consistency | 97,48 | 98,78 | 94,26 | 99,05 | 93,71 | 91,62 | 90,04 | 94,58 | 89,78 | 94,49 |
| CONSS (dual-level) | 97,66 | 98,95 | 94,31 | 98,89 | 94,80 | 92,65 | 94,09 | 96,52 | 90,73 | 95,04 |

As colunas das classes trazem a acurácia por classe (recall), não o IoU.

### 3.2 Pham et al. (2026), `2_Pham2026_Voxel3D_Facies_arXiv.pdf`

A comparação mais próxima pela arquitetura (3D por voxel no mesmo volume).

- **Referência:** D.-T. Pham, M.-T. Pham, A. Nguyen, V. Nguyen. *Voxel-based 3D Facies Segmentation from Seismic Data:
  A Comparative Study*. arXiv:2608.14058, 2026 (preprint, sem revisão por pares).
- **Dados:** o F3 de Alaudah `601 × 901 × 255` e o Parihaka.
- **Divisão:**
  - validação cruzada em 5 dobras de 80/20, seguindo Canguçu et al. (2025);
  - cada dobra tem `601 × 180 × 255`, ou seja, pela forma, **blocos contíguos de 180 xlines** (o texto não diz o eixo);
  - os cubos de 128³ (passo 64, borda espelhada) são cortados dentro de cada dobra: 975 de treino e 195 de validação;
  - não há teste separado; a métrica é a da dobra de validação.
- **Redes:** 3D U-Net (2,51 M parâmetros), UNETR (146,6 M) e SegMamba (67,4 M). Os hiperparâmetros "seguem trabalhos
  anteriores" e não são informados.
- **Métrica:** Dice por classe (%) e a média.
- **Resultados (Tabela II(a)):**

| Modelo | Upper | Middle | Lower | R/C | Scruff | Zechstein | Média |
|---|---|---|---|---|---|---|---|
| 3D U-Net | 96,67 | 88,02 | 96,40 | 88,09 | 77,07 | 79,06 | 87,69* |
| UNETR | 93,08 | 73,08 | 95,40 | 88,84 | 75,80 | 72,72 | 83,16 |
| SegMamba | 96,05 | 75,94 | 95,97 | 80,29 | 75,43 | 81,57 | 84,21 |

\* A média das seis classes listadas dá 87,55, não 87,69.

A Tabela IV mostra a sensibilidade da 3D U-Net ao tamanho do cubo: 96³ dá 83,63, 128³ dá 87,69 e 144³ dá 85,04.

### 3.3 Alaudah et al. (2019), `3_Alaudah2019_Benchmark_F3_arXiv.pdf`

A referência que define os dados, a divisão oficial e as métricas.

- **Referência:** Y. Alaudah, P. Michałowicz, M. Alfarraj, G. AlRegib. *A machine-learning benchmark for facies
  classification*. Interpretation 7(3), SE175–SE187, 2019. doi:10.1190/INT-2018-0249.1. Preprint em arXiv:1901.07659.
- **Divisão oficial (extrapolação):**
  - treino: inlines 300–700 × xlines 300–1000;
  - teste 1: inlines 100–299 × xlines 300–1000;
  - teste 2: inlines 100–700 × xlines 1001–1200, com um diápiro de Zechstein que o treino nunca vê.
- **Aviso dos autores:** sortear as seções de treino e de teste "leva a resultados de teste artificialmente bons".
- **Redes:** deconvnet 2D por patch e por seção (+ aumentação, + *skip*).
- **Métricas:** PA, acurácia por classe, MCA e FWIU, nos dois testes juntos. O artigo não dá mIoU.
- **Resultados (Tabela 2):**

| Modelo | PA | Upper | Middle | Lower | R/C | Scruff | Zechstein | MCA | FWIU |
|---|---|---|---|---|---|---|---|---|---|
| Patch | 0,788 | 0,754 | 0,804 | 0,992 | 0,499 | 0,074 | 0,264 | 0,565 | 0,640 |
| Patch + aug + skip | 0,862 | 0,926 | 0,912 | 0,974 | 0,673 | 0,286 | 0,458 | 0,705 | 0,757 |
| Seção | 0,879 | 0,973 | 0,872 | 0,951 | 0,744 | 0,539 | 0,219 | 0,716 | 0,789 |
| Seção + aug | 0,901 | 0,956 | 0,940 | 0,979 | 0,812 | 0,423 | 0,714 | 0,804 | 0,844 |
| Seção + aug + skip | 0,905 | 0,974 | 0,938 | 0,941 | 0,772 | 0,674 | 0,602 | 0,817 | 0,832 |

### 3.4 Civitarese et al. (2019), `4_Civitarese2019_Semantic_Segmentation_Seismic_arXiv.pdf`

O análogo do protocolo `step`, mas em outro dado.

- **Referência:** D. Civitarese, D. Szwarcman, E. Vital Brazil, B. Zadrozny. *Semantic Segmentation of Seismic
  Images*. arXiv:1905.04307, 2019 (IBM Research).
- **Dados:** **Penobscot**, não o F3. São 459 inlines depois da limpeza e 7 classes.
- **Divisão:**
  - o cubo é dividido em `n` blocos de inlines;
  - em cada bloco, os primeiros 70 % vão para o treino e o resto para a validação;
  - um cenário limitado sorteia `x/n` seções por bloco, com x = 5, 9, 13 ou 100;
  - o teste são 40 imagens do mesmo cubo (o artigo não detalha a escolha).
- **Métrica:** mmIoU, a média por imagem do mIoU da imagem.
- **Resultados (Tabela 2, tile 80 × 120, mmIoU):**

| Seções de treino | 5 | 9 | 13 | 100 |
|---|---|---|---|---|
| DanetFCN3 | 0,8385 | 0,9689 | 0,9764 | 0,9900 |
| DanetFCN | 0,9180 | 0,9592 | 0,9700 | 0,9877 |
| FCN8 | 0,5819 | 0,7196 | 0,7389 | 0,9364 |

Esses números mostram o quanto a interpolação dentro do mesmo cubo é fácil: com 9 seções de 459, o mmIoU já passa de 0,96.

### 3.5 Li et al. (2026), `5_Li2026_Leakage_Aware_Frontiers.pdf`

A medida do vazamento por vizinhança e da eficiência de rótulo.

- **Referência:** Z. Li, J. Liu, X. Li, T. He, K. Li, Y. Zhang, Z. Pan, X. Lei, S. Wang, Z. Zhang. *Leakage-aware
  full-section evaluation and label-efficient semi-supervised seismic facies segmentation for subsurface geological
  modeling*. Frontiers in Earth Science 14:1910345, 2026. doi:10.3389/feart.2026.1910345 (CC-BY).
- **Protocolo:**
  - no F3, divisão oficial de Alaudah, seções inteiras, treino e validação em intervalos contíguos com folga e teste
    sempre nos cubos test1 e test2;
  - 11 redes 2D, 3 sementes;
  - métricas PA, MCA, mIoU, FWIU e IoU por classe.
- **F3 (test1 mIoU):**
  - CNNs treinadas do zero ficam em 0,709–0,751;
  - SegFormer-B0 pré-treinada: 0,755;
  - U-Net/ResNet34 pré-treinada: 0,780;
  - SegFormer-B2: 0,787;
  - a melhor configuração dá PA/MCA/FWIU 0,919/0,891/0,856, contra 0,905/0,817/0,832 de Alaudah.
- **Eficiência de rótulo (SegFormer-B0 supervisionada, test1 mIoU):** 5 % dá 0,686, 10 % dá 0,716, 25 % dá 0,740,
  50 % dá 0,749 e 100 % dá 0,755.
- **Vazamento por vizinhança (Parihaka, 5 % de rótulos, Tabela 7):**
  - numa divisão aleatória de inlines, ~170 das 201 inlines de teste têm uma de treino a 1 inline;
  - forçando uma folga mínima de 1, 6 e 11 inlines, o mIoU supervisionado cai de **0,474 para 0,442 e 0,412**;
  - os autores só usam a divisão aleatória para comparar com a literatura.

### 3.6 SeisSegDiff: Ore & Gao (2025), `6_Ore2025_SeisSegDiff_CAGEO.pdf`

O segundo estudo com o mesmo protocolo do `step` no F3.

- **Referência:** T. Ore, D. Gao. *SeisSegDiff: A label-efficient few-shot texture segmentation diffusion model for
  seismic facies classification*. Computers & Geosciences 196, 105823, 2025. doi:10.1016/j.cageo.2024.105823. O PDF
  é o manuscrito aceito, de acesso público no NSF PAR (par.nsf.gov/servlets/purl/10566315).
- **Dados:** o F3 com a interpretação de Alaudah (6 classes; 601 inlines × 901 xlines) e o Parihaka. O artigo fala em
  "872 depth points", não nas 255 amostras do volume do benchmark, e não explica a diferença.
- **Divisão (duas estratégias, as duas com 5 inlines de treino):**
  - **Estratégia 1:** 5 inlines do cubo de treino; o teste são as 200 inlines do oeste (30 % do volume).
  - **Estratégia 2:** 5 inlines **igualmente espaçadas** no volume inteiro (< 1 %, uma a cada ~120); **o teste é o
    resto do volume**. É o mesmo desenho do `step` e do CONSS, com a inline de teste a até ~60 inlines do rótulo.
- **Rede:** representações de uma U-Net de difusão (2D, pré-treinada para gerar sísmica), um classificador por pixel
  em cima e um ensemble de 10 modelos.
- **Métricas:** acurácia, precisão, recall e IoU por classe, e as médias (inclusive F1).
- **Resultados no F3 (Tabela 2):**

| Estratégia | Upper | Middle | Lower | R/C | Scruff | Zechstein | mIoU | Recall médio | F1 médio |
|---|---|---|---|---|---|---|---|---|---|
| 2 (intercalada), IoU | 96,1 | 83,9 | 96,5 | 75,7 | 78,8 | 53,4 | 80,7 | 87,5 | 89 |
| 2 (intercalada), recall | 97,7 | 91,8 | 98,3 | 82,7 | 93,3 | 61,4 | | | |
| 1 (cubo separado), IoU | 94,5 | 79,3 | 93,6 | 69,2 | 60,1 | 49,1 | 74,3 | 82,7 | 84 |

A coluna de "Accuracy" por classe do artigo (~0,98) não é a acurácia por classe de Alaudah: ela conta os verdadeiros
negativos. Por isso usei o recall, que é a CA.

## 4. Resultados deste repositório fora do `step`

Rodei o `3 - Predict` novo nos dois modelos: no volume real, sem a borda. Todos os valores estão em %.

| Modelo | PA | MCA | mIoU | FWIU | F1 médio |
|---|---|---|---|---|---|
| model_1 (DBRNet) | 97,71 | 94,78 | 90,88 | 95,60 | 95,13 |
| model_2 (ResACE-UNet GRVA) | 97,66 | 94,46 | 90,51 | 95,50 | 94,90 |

**Por classe**

| Classe | CA m1 | CA m2 | IoU m1 | IoU m2 | F1 m1 | F1 m2 |
|---|---|---|---|---|---|---|
| 0 Upper N.S. | 99,12 | 99,09 | 97,97 | 97,83 | 98,97 | 98,90 |
| 1 Middle N.S. | 95,22 | 95,35 | 90,84 | 90,88 | 95,20 | 95,22 |
| 2 Lower N.S. | 98,91 | 98,89 | 97,73 | 97,67 | 98,85 | 98,82 |
| 3 Rijnland/Chalk | 93,33 | 93,37 | 88,65 | 88,18 | 93,98 | 93,72 |
| 4 Scruff | 93,97 | 93,66 | 88,31 | 88,85 | 93,79 | 94,10 |
| 5 Zechstein | 88,13 | 86,37 | 81,78 | 79,63 | 89,98 | 88,66 |

**Efeito de tirar a borda refletida (`block_iou` médio)**

| Modelo | Antes (volume `603 × 287 × 933`) | Depois (região real) |
|---|---|---|
| model_1 | 90,58 | 90,88 |
| model_2 | 89,89 | 90,51 |

Sem a borda, a classe 0 cai um pouco: a faixa de cima refletida era quase toda Upper N.S., que é fácil. As classes 3 e
4 sobem.

**Métrica por distância até a inline de treino mais próxima**

Análise feita só nesta rodada, fora do notebook. Cada distância tem 100 inlines.

| Distância | mIoU m1 | mIoU m2 | PA m2 |
|---|---|---|---|
| 1 | 90,77 | 90,40 | 97,63 |
| 2 | 90,98 | 90,66 | 97,70 |
| 3 | 90,98 | 90,63 | 97,69 |
| 4 | 90,79 | 90,33 | 97,62 |

**Leitura:** de 1 a 4 inlines a métrica não muda. A pequena variação acompanha a janela de 4 inlines: as distâncias 1 e
4 caem na borda do bloco predito, e a 2 e a 3 no meio. Ou seja, o protocolo daqui não mede nada além de 4 inlines de
distância, e é aí que o CONSS e os outros medem.

**Teste em tiles (`test_iou`, só referência)**

| Modelo | PA | MCA | mIoU | FWIU | F1 médio |
|---|---|---|---|---|---|
| model_1 | 94,89 | 90,35 | 82,68 | 90,72 | 89,85 |
| model_2 | 95,01 | 89,66 | 83,07 | 90,86 | 90,29 |

Esse teste refaz o `test_iou` gravado no treino (82,68 e 83,07), o que confirma que a conta nova bate com a antiga.

## 5. Tabelas de comparação

### 5.1 Médias

| Estudo / modelo | Rede | Rótulos de treino | Onde mede | Distância ao rótulo | PA | MCA | mIoU | FWIU | F1/Dice |
|---|---|---|---|---|---|---|---|---|---|
| **model_1** | DBRNet 3D, 4×64×64 | 201 de 601 inlines (33 %), 4 a cada 12 | as 400 inlines restantes | 1–4 inlines | 97,71 | 94,78 | 90,88 | 95,60 | 95,13 |
| **model_2** | ResACE-UNet 3D, 4×64×64 | idem | idem | 1–4 inlines | 97,66 | 94,46 | 90,51 | 95,50 | 94,90 |
| CONSS, supervisionado | DeepLabv3+ 2D, 256×256 | 7 inlines (1,2 %), 1 a cada ~86 | as 594 inlines restantes | 1–~43 (~85 no fim) | 96,62 | 92,09 | 87,12 | – | 92,87 |
| CONSS, semissupervisionado | DeepLabv3+ 2D + contraste | 7 inlines + sísmica sem rótulo das 594 | idem | idem | 97,66 | 96,52 | 90,73 | – | 95,04 |
| SeisSegDiff, estratégia 2 | difusão 2D + classificador | 5 inlines (< 1 %), 1 a cada ~120 | o resto do volume | 1–~60 | – | 87,5 (recall) | 80,7 | – | 89 |
| SeisSegDiff, estratégia 1 | idem | 5 inlines do cubo de treino | 200 inlines do oeste | 1–200 | – | 82,7 (recall) | 74,3 | – | 84 |
| Pham, 3D U-Net | 3D, 128³ | 4 de 5 blocos de 180 xlines (80 %) | o bloco de 180 xlines de fora | 1–90 (1–180 nas pontas) | – | – | – | – | 87,69 |
| Pham, SegMamba | 3D, 128³ | idem | idem | idem | – | – | – | – | 84,21 |
| Pham, UNETR | 3D, 128³ | idem | idem | idem | – | – | – | – | 83,16 |
| Alaudah, seção + aug + skip | deconvnet 2D | cubo oficial (~52 % do volume) | test1 + test2 (regiões vizinhas) | 1–200 | 90,5 | 81,7 | – | 83,2 | – |
| Li 2026, melhor configuração | 2D, seção | cubo oficial | test1 | 1–200 | 91,9 | 89,1 | 78,0–78,7 | 85,6 | – |
| Li 2026, 5 % dos rótulos | SegFormer-B0 2D | 5 % das seções do cubo oficial | test1 | 1–200 | – | – | 68,6 | – | – |
| Civitarese, DanetFCN3 (Penobscot) | 2D, 80×120 | 9 seções de 459 (2 %) | 40 seções do mesmo cubo | não informado | – | – | 96,9 (mmIoU) | – | – |

### 5.2 Acurácia por classe (recall) contra o CONSS, o SeisSegDiff e Alaudah (%)

| Classe | model_1 | model_2 | CONSS sup. | CONSS | SeisSegDiff (estr. 2) | Alaudah (seção + aug + skip) |
|---|---|---|---|---|---|---|
| 0 Upper N.S. | 99,12 | 99,09 | 98,38 | 98,95 | 97,7 | 97,4 |
| 1 Middle N.S. | 95,22 | 95,35 | 93,27 | 94,31 | 91,8 | 93,8 |
| 2 Lower N.S. | 98,91 | 98,89 | 99,14 | 98,89 | 98,3 | 94,1 |
| 3 Rijnland/Chalk | 93,33 | 93,37 | 89,41 | 94,80 | 82,7 | 77,2 |
| 4 Scruff | 93,97 | 93,66 | 92,70 | 92,65 | 93,3 | 67,4 |
| 5 Zechstein | 88,13 | 86,37 | 79,53 | 94,09 | 61,4 | 60,2 |
| MCA | 94,78 | 94,46 | 92,09 | 96,52 | 87,5 | 81,7 |

### 5.2b IoU por classe contra o SeisSegDiff (%)

O SeisSegDiff é o único estudo do mesmo protocolo que publica o IoU por classe.

| Classe | model_1 | model_2 | SeisSegDiff (estr. 2) |
|---|---|---|---|
| 0 Upper N.S. | 97,97 | 97,83 | 96,1 |
| 1 Middle N.S. | 90,84 | 90,88 | 83,9 |
| 2 Lower N.S. | 97,73 | 97,67 | 96,5 |
| 3 Rijnland/Chalk | 88,65 | 88,18 | 75,7 |
| 4 Scruff | 88,31 | 88,85 | 78,8 |
| 5 Zechstein | 81,78 | 79,63 | 53,4 |
| mIoU | 90,88 | 90,51 | 80,7 |

### 5.3 Dice/F1 por classe contra Pham et al. (%)

| Classe | model_1 | model_2 | 3D U-Net | SegMamba | UNETR |
|---|---|---|---|---|---|
| 0 Upper N.S. | 98,97 | 98,90 | 96,67 | 96,05 | 93,08 |
| 1 Middle N.S. | 95,20 | 95,22 | 88,02 | 75,94 | 73,08 |
| 2 Lower N.S. | 98,85 | 98,82 | 96,40 | 95,97 | 95,40 |
| 3 Rijnland/Chalk | 93,98 | 93,72 | 88,09 | 80,29 | 88,84 |
| 4 Scruff | 93,79 | 94,10 | 77,07 | 75,43 | 75,80 |
| 5 Zechstein | 89,98 | 88,66 | 79,06 | 81,57 | 72,72 |
| Média | 95,13 | 94,90 | 87,69 | 84,21 | 83,16 |

### 5.4 Leitura

- **Contra o CONSS (o par mais justo):**
  - os modelos daqui superam o CONSS supervisionado em PA, MCA, mIoU e F1, e empatam com o CONSS semissupervisionado
    no mIoU (90,9 e 90,5 contra 90,7);
  - mas usam ~29× mais inlines rotuladas, e a inline de teste mais distante fica a 4 inlines, não a ~43;
  - com os dois protocolos tão diferentes, a diferença de 3–4 pontos não indica qual método é melhor;
  - o padrão por classe é o mesmo: Zechstein e Rijnland/Chalk são as classes difíceis em todos os trabalhos.
- **Contra o SeisSegDiff:** mesmo protocolo, com 5 inlines e distância de até ~60. A diferença de 10 pontos de mIoU
  (90,9 contra 80,7) cai quase toda nas classes raras (Zechstein: 81,8 contra 53,4). É exatamente onde mais rótulo e
  mais vizinhança ajudam, então ela também não separa a rede do protocolo.
- **Contra Pham et al.:** o Dice médio daqui (95,1) passa a 3D U-Net (87,7). Porém a validação deles é um bloco
  contíguo de 180 xlines, ou seja, interpolação ou extrapolação por 90 a 180 traços, bem mais difícil que 8 inlines.
  Os números não são comparáveis diretamente.
- **Contra Alaudah e Li (divisão oficial):** a divisão deles é extrapolação para regiões vizinhas. Os 75–79 % de mIoU
  deles contra os 91 % daqui refletem sobretudo o protocolo, não a rede.

## 6. Ressalvas

1. **Proximidade espacial (a principal).**
   - Aqui, toda inline de teste está a 1–4 inlines de uma inline de treino.
   - As camadas do F3 são lateralmente contínuas: entre blocos de treino 8 inlines à parte, a tarefa é quase
     interpolação.
   - Li et al. (2026) mostram que tirar a vizinhança derruba o supervisionado de 0,474 para 0,412 de mIoU com folga de
     11 inlines. Alaudah et al. (2019) avisam que divisões sorteadas dão resultados "artificialmente bons".
   - A análise por distância daqui mostra que, dentro de 1–4 inlines, a distância não pesa. Logo, **os números são do
     regime de interpolação curta** e não dizem como a rede se comporta a dezenas de inlines do rótulo.
   - **Isso afeta o `block_iou`, não só o `test_iou`.** São dois problemas diferentes:
     - **`test_iou` (vazamento):** os tiles de teste saem das **mesmas inlines** dos tiles de treino, só que sorteados.
       Sortear não protege nada: o tile de teste tem os vizinhos laterais e as 4 inlines dele no treino.
     - **`block_iou` (proximidade):** as inlines de teste **nunca foram vistas**, então não há vazamento e a métrica é
       honesta. Mas "não visto" não quer dizer "longe": cada inline de teste está a 1–4 inlines de uma de treino. O
       `block_iou` mede bem a capacidade de interpolar a essa distância, e só essa.
   - **A prova está nos backups do Marlim** (`Documents/Backups/Backup2`). É a mesma `resaceunet_grva` `4×256×256`, com
     a mesma loss, lr, lote e 200 épocas; só muda o `step`:

   | Backup | `step` | Inlines de treino | Distância máx. ao rótulo | `test_iou` | `block_iou` |
   |---|---|---|---|---|---|
   | Backup2/model_0 e model_3 | 3 | 33 % | 4 | 93,2 / 93,0 | 95,7 / 95,4 |
   | Backup2/model_2 | 5 | 20 % | 8 | 89,0 | 93,5 |
   | Backup2/model_1 | 10 | 10 % | 18 | 84,0 | 87,6 |

   O `block_iou` cai 8 pontos sem mudar nada além da quantidade de rótulo e da distância. Se ele medisse uma
   generalização independente disso, ficaria parado. No F3, com os estudos a 43–60 inlines de distância, a queda tende
   a ser maior ainda.

2. **Orçamento de rótulos.** São 33 % das inlines (≈ 27 % dos voxels no gradiente) contra 1,2 % do CONSS, < 1 % do
   SeisSegDiff, 80 % de Pham, ~52 % de Alaudah e 5–100 % de Li. Só o CONSS e o SeisSegDiff têm o mesmo desenho, e com
   29–40× menos rótulo.
3. **Contexto 2D × 3D.**
   - O tile `4 × 64 × 64` tem pouco contexto lateral e de profundidade: 64 amostras de 255 e 64 xlines.
   - O CONSS e Alaudah usam janelas de 256 ou seções inteiras, e Pham, cubos de 128³.
   - Alaudah mostra que redes de patch confundem classes pela profundidade, e as de seção não. Os modelos daqui só vão
     bem porque, a 1–4 inlines, o vizinho rotulado tem a mesma geometria.
4. **O CONSS semissupervisionado é transdutivo.** Ele vê a sísmica (sem rótulo) das inlines de teste no treino. O
   par direto dos modelos daqui, que são supervisionados, é o CONSS **supervisionado** (mIoU 87,12).
5. **Agregação da métrica.** Aqui, no CONSS, em Alaudah e em Li 2026 tudo é somado por pixel ou voxel antes de dividir.
   Pham calcula o Dice por dobra (a forma exata da média não é descrita). O mmIoU de Civitarese é por imagem. As médias
   são simples sobre as 6 classes, exceto a FWIU.
6. **A borda refletida estava dentro da métrica antes.** O `block_iou` antigo dos modelos 1 e 2 media o volume
   `603 × 287 × 933`, com ~14 % de voxels repetidos da borda. Agora o `3 - Predict` mede só o `VALID`: o mIoU subiu
   0,3 (m1) e 0,6 (m2) ponto. Os números da seção 4 já são os novos.
7. **O `test_iou` não serve para comparar.**
   - Os tiles de teste são sorteados entre os tiles das mesmas inlines do treino.
   - Eles também incluem a borda que o `Format` completa até múltiplo de 64: 33 amostras no fundo e 27 xlines. Nessa
     borda a sísmica é refletida, mas o rótulo é a última linha repetida (`mode='edge'`).
   - Provavelmente é por isso que o `test_iou` (82,7 e 83,1) fica abaixo do volume (90,9 e 90,5), com a Zechstein, no
     fundo, como a classe mais afetada (IoU 56,1/64,2 nos tiles contra 81,8/79,6 no volume). Isso não foi verificado
     isoladamente.
8. **Uma semente por modelo.** Li et al. relatam desvio de ~0,01 de mIoU entre sementes. A diferença entre o model_1 e
   o model_2 (0,4 ponto) está dentro desse ruído.
9. **Pham et al. é preprint.** Não informa os hiperparâmetros e tem a média da 3D U-Net inconsistente (87,69 contra
   87,55 recalculado).
10. **Pré-processamento.**
    - Os patches do F3 daqui chegam em [0, 1] e não recebem nenhuma normalização (`scaling` null).
    - O CONSS não informa a normalização.
    - Li padroniza pela média e pelo desvio do cubo de treino.
    - Todos usam o mesmo rótulo de Alaudah, que vem de um modelo geológico com artefatos de conversão tempo-profundidade
      (citados pelos autores).

## 7. Como fazer a comparação justa

O `step` do `Format` controla exatamente a fração de treino e a distância do teste. No F3 (região real, blocos de 4
inlines):

| `step` | Inlines de treino | Inlines de teste | Distância máx. / média | Estudo no mesmo regime |
|---|---|---|---|---|
| 3 (model_1 e model_2) | 201 (33,4 %) | 400 | 4 / 2,5 | nenhum |
| 5 | 121 (20,1 %) | 480 | 8 / 4,5 | nenhum |
| 10 | 61 (10,1 %) | 540 | 18 / 9,5 | nenhum |
| 25 | 25 (4,2 %) | 576 | 48 / 24,5 | CONSS (7 inlines, até ~43) |
| 30 | 21 (3,5 %) | 580 | 58 / 29,5 | SeisSegDiff (5 inlines, até ~60) |

Treinar a mesma rede no F3 com `step` 3, 5, 10, 25 e 30 dá uma curva de mIoU por fração de rótulo. As pontas de 25 e
30 caem no regime dos dois estudos.

1. **Regime do CONSS (pronto para rodar, sem mudar código):**
   - Use uma rodada com `"step": 25` no `Task/task.json`.
   - O `Format` corta um bloco de 4 inlines a cada 100: 7 blocos (as mesmas 7 posições do CONSS), 25 inlines reais
     (4,2 %) e 525 tiles.
   - O `3 - Predict` mede as 576 inlines restantes, a até 48 inlines do rótulo.
   - Compare com o CONSS **supervisionado** (PA 96,62, MCA 92,09, mIoU 87,12, F1 92,87).
   - Ainda são 4 inlines por posição contra 1, mas é o mesmo regime de distância. Custa um treino por rede.
   - **Regime do SeisSegDiff:** a mesma ideia com `"step": 30` (6 blocos a cada 120 inlines), comparando com a
     estratégia 2 (mIoU 80,7, IoU por classe na seção 5.2b).
2. **Regime de Pham et al.:** 5 dobras espaciais de 180 xlines contíguas (no `VALID`, xlines 16–916). Precisa de uma
   opção nova no `Format`/`0 - Model_CV` para dividir por região em vez de sortear tiles.
3. **Regime oficial (Alaudah e Li):**
   - Treine em inlines 300–700 × xlines 300–1000 e teste nos cubos test1 e test2.
   - No `VALID`, a inline `i` corresponde à inline `99 + i` e a xline `j` à xline `284 + j`, supondo a ordem crescente
     dos eixos do benchmark.
   - Também precisa de recorte por região no `Format` e no `3 - Predict`.
4. **Variância:** `n_trials ≥ 3` em qualquer um dos regimes, para dar média ± desvio como Li et al.

## Referências

1. Li, K., Liu, W., Dou, Y., Xu, Z., Duan, H., Jing, R. (2023). CONSS: Contrastive Learning Method for Semisupervised
   Seismic Facies Classification. *IEEE JSTARS* 16, 7838–7849. doi:10.1109/JSTARS.2023.3308754. Arquivo:
   `1_Li2023_CONSS_JSTARS.pdf`.
2. Pham, D.-T., Pham, M.-T., Nguyen, A., Nguyen, V. (2026). Voxel-based 3D Facies Segmentation from Seismic Data: A
   Comparative Study. arXiv:2608.14058. Arquivo: `2_Pham2026_Voxel3D_Facies_arXiv.pdf`.
3. Alaudah, Y., Michałowicz, P., Alfarraj, M., AlRegib, G. (2019). A machine-learning benchmark for facies
   classification. *Interpretation* 7(3), SE175–SE187. doi:10.1190/INT-2018-0249.1 (arXiv:1901.07659). Arquivo:
   `3_Alaudah2019_Benchmark_F3_arXiv.pdf`.
4. Civitarese, D., Szwarcman, D., Vital Brazil, E., Zadrozny, B. (2019). Semantic Segmentation of Seismic Images.
   arXiv:1905.04307. Arquivo: `4_Civitarese2019_Semantic_Segmentation_Seismic_arXiv.pdf`.
5. Li, Z., Liu, J., Li, X., He, T., Li, K., Zhang, Y., Pan, Z., Lei, X., Wang, S., Zhang, Z. (2026). Leakage-aware
   full-section evaluation and label-efficient semi-supervised seismic facies segmentation for subsurface geological
   modeling. *Frontiers in Earth Science* 14:1910345. doi:10.3389/feart.2026.1910345. Arquivo:
   `5_Li2026_Leakage_Aware_Frontiers.pdf`.
6. Canguçu, G. L. et al. (2025). A Principled Benchmark for Seismic Data Segmentation. *IEEE TGRS* 63 (sem PDF aberto).
7. Gutierrez, G. B. et al. (2025). On the Performance Evaluation of Deep Learning Models for Seismic Facies
   Segmentation. *Geophysical Prospecting*. doi:10.1111/1365-2478.70104 (CC-BY; baixar pelo navegador).
8. Abid, B., Khan, B. M., Memon, R. A. (2022). Seismic Facies Segmentation Using Ensemble of Convolutional Neural
   Networks. *Wireless Communications and Mobile Computing*. doi:10.1155/2022/7762543.
9. Ore, T., Gao, D. (2025). SeisSegDiff: A label-efficient few-shot texture segmentation diffusion model for seismic
   facies classification. *Computers & Geosciences* 196, 105823. doi:10.1016/j.cageo.2024.105823. Arquivo:
   `6_Ore2025_SeisSegDiff_CAGEO.pdf`.
10. Tolstaya, E., Egorov, A. (2022). Deep learning for automated seismic facies classification. *Interpretation*
    10(2). doi:10.1190/INT-2021-0140.1.
