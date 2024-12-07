import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import os
from art.estimators.classification import PyTorchClassifier
import timm
from timm import create_model

import warnings
import pytorch_fid.fid_score as fid_score

warnings.filterwarnings("ignore")

def get_vim_model(type="small"):
    import sys

    sys.path.append('/Dir to the VisionMamba code of/Vim-main/vim')

    import models_mamba
    if type == "small":
        model_name = 'vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2'
    elif type == "tiny":
        model_name = 'vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2'
    else:
        assert False
    print("Creating Vim")
    model = create_model(
        model_name=model_name,
        pretrained=False,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size=224
    )
    if type == "small":
        path = "Dir to the VisionMamba code of Vim-main/vim/ckpts/vim_s_midclstok_80p5acc.pth"
    elif type == "tiny":
        path = "Dir to the VisionMamba code of Vim-main/vim/ckpts/vim_t_midclstok_76p1acc.pth"
    else:
        assert False
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.cuda()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    return model

def model_selection(name):
    if name == "resnet50":
        model = models.resnet50(pretrained=True)
    elif name == "convnext":
        model = models.convnext_base(pretrained=True)
    elif name == "swin":
        model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
    elif name == "vim-small":
        model = get_vim_model("small")
    elif name == "vim-tiny":
        model = get_vim_model("tiny")
    elif name == "vgg19":
        model = models.vgg19(pretrained=True)
    elif name == "mobile_v2":
        model = models.mobilenet_v2(pretrained=True)
    elif name == "wrn50":
        model = models.wide_resnet50_2(pretrained=True)

    else:
        raise NotImplementedError("No such model!")
    return model.cuda()

if __name__ == "__main__":
    model = model_selection("vim-tiny")
    print(model)