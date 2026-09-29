import torch.nn.functional as F
import torch
import torch.optim as optim
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.classification import BinaryJaccardIndex
from monai.networks.nets import SegResNet

from utils.index import getDevices
from .types.UNet3D import UNet3D
from .types.Unet3D_V2 import Unet3D_V2
from .types.resaceunet import ResACEUnet


class ModelNetwork:
    selected = None

    def __init__(self, network, img_size, classes=1, channels=1, lr=1e-4, dropout=0.1, num_filters=16, deep_supervision=False, gpu_ids=None):
        self.network = network
        self.img_size = img_size
        self.classes  = classes
        self.multiclass = (self.classes > 1)
        self.channels   = channels
        self.dropout    = dropout
        self.num_filters = num_filters
        self.deep_supervision = deep_supervision   # só as redes ResACE suportam; ignorado pelas demais
        self.lr = lr
        
        self.gpu_ids = list(gpu_ids) if gpu_ids is not None else getDevices(verbose=False)
        self.device  = torch.device(f'cuda:{self.gpu_ids[0]}') if self.gpu_ids else torch.device('cpu')

        # module = modelo puro (pesos, save/load) | model = o que roda o forward,
        # replicado nas demais GPUs quando mais de uma foi selecionada
        self.module = self.get().to(self.device)
        self.model  = torch.nn.DataParallel(self.module, device_ids=self.gpu_ids) if len(self.gpu_ids) > 1 else self.module
        self.optimizer = optim.AdamW(self.module.parameters(), lr=self.lr, weight_decay=1e-4)

        if self.multiclass:
            self.iou = MulticlassJaccardIndex(num_classes=self.classes, average='macro').to(self.device)
        else:
            self.iou = BinaryJaccardIndex(threshold=0.5).to(self.device)
    
    def get(self):
        classes = self.classes

        if self.network == 'unet':
            return UNet3D(img_channels=self.channels, num_filters=self.num_filters, dropout=self.dropout, classes=classes)
        
        if self.network == 'unet3d_v2':
            return Unet3D_V2(img_channels=self.channels, classes=classes, num_filters=self.num_filters, dropout=self.dropout)

        if self.network == 'segresnet':
            return SegResNet(spatial_dims=3, in_channels=self.channels, out_channels=self.classes, init_filters=self.num_filters, dropout_prob=self.dropout)
        
        if self.network == 'resaceunet':
            return ResACEUnet(in_channels=self.channels, num_classes=classes, base_filters=self.num_filters, dropout_rate=self.dropout, deep_supervision=self.deep_supervision, input_shape=self.img_size)

        return None