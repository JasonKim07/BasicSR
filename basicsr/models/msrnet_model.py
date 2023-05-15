import numpy as np
import random
import torch
from torch.nn import functional as F

from basicsr.archs import build_network
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.sr_model import SRModel
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register(suffix='basicsr')
class MultiFeatureSRNetModel(SRModel):
    """MultiFeatureSRNet Model: Training

    It is trained without GAN losses.
    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(MultiFeatureSRNetModel, self).__init__(opt)

        # load pretrained models (RRDBNet)
        self.rrdbnet = build_network(opt['network_g_pre'])
        load_path = self.opt['path_pre'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path_pre'].get('param_key_g', 'params')
            self.load_network(self.rrdbnet, load_path, self.opt['path_pre'].get('strict_load_g', True), param_key)

        # Freeze a RRDB layer
        for name, param in self.rrdbnet.named_parameters():
            if name.startswith("body"):
                param.requires_grad = False

        self.net_g.RRDB = self.rrdbnet.body

        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(MultiFeatureSRNetModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True
