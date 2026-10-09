import torch.nn.functional as F
import torch
import torch.optim as optim
from monai.networks.nets import SegResNet


# SEGRESNET DO MONAI COM PAD: OS len(blocks_down) - 1 DOWNSAMPLES DIVIDEM D, H E W POR 2, E O TILE (4, 256, 256) QUEBRAVA
# NO DECODER (D DESCE 4 -> 2 -> 1 -> 1 E SOBE 1 -> 2 -> 4, QUE NAO BATE COM O SKIP DE 2). PREENCHE COM ZEROS ATE MULTIPLO
# DE 8 E CORTA NA SAIDA. SUBCLASSE, E NAO WRAPPER, PARA O state_dict TER AS MESMAS CHAVES DA SegResNet ORIGINAL
class Segresnet(SegResNet):
    def __init__(self, channels, classes, num_filters, dropout):
        super().__init__(spatial_dims=3, in_channels=channels, out_channels=classes, init_filters=num_filters, dropout_prob=dropout)
        self.multiple = 2 ** (len(self.blocks_down) - 1)

    def forward(self, x):
        d, h, w = x.shape[2:]
        pad_d = (self.multiple - d % self.multiple) % self.multiple
        pad_h = (self.multiple - h % self.multiple) % self.multiple
        pad_w = (self.multiple - w % self.multiple) % self.multiple

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))

        return super().forward(x)[:, :, :d, :h, :w]
