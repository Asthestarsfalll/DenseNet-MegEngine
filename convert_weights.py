import argparse
import os

import megengine as mge
import numpy as np
import torch
import torch.nn as nn

from models.densenet import densenet121, densenet161, densenet169, densenet201
from models.torch_model import densenet121 as torch_densenet121 
from models.torch_model import densenet161 as torch_densenet161
from models.torch_model import densenet169 as torch_densenet169
from models.torch_model import densenet201 as torch_densenet201
from models.torch_model import model_urls


def get_atttr_by_name(torch_module, k):
    name_list = k.split('.')
    sub_module = getattr(torch_module, name_list[0])
    if len(name_list) != 1:
        for i in name_list[1:-1]:
            try:
                sub_module = getattr(sub_module, i)
            except:
                sub_module = sub_module[int(i)]
    return sub_module


def convert(torch_model, torch_dict):
    new_dict = {}
    for k, v in torch_dict.items():
        data = v.numpy()
        sub_module = get_atttr_by_name(torch_model, k)
        is_conv = isinstance(sub_module, nn.Conv2d)
        if is_conv:
            groups = sub_module.groups
            is_group = groups > 1
        else:
            is_group = False
        if "weight" in k and is_group:
            out_ch, in_ch, h, w = data.shape
            data = data.reshape(groups, out_ch // groups, in_ch, h, w)
        if "bias" in k:
            if is_conv:
                data = data.reshape(1, -1, 1, 1)
        if "num_batches_tracked" in k:
            continue
        new_dict[k] = data
    return new_dict


def main(torch_name):
    url = model_urls[torch_name]
    # download manually if speed is too slow
    # torch_state_dict = torch.hub.load_state_dict_from_url(
    #     url, map_location='cpu', progress=True, check_hash=True)
    torch_model = eval("torch_" + torch_name)(True)
    torch_state_dict = torch_model.state_dict()
    model = eval(torch_name)()

    new_dict = convert(torch_model, torch_state_dict)

    model.load_state_dict(new_dict)
    os.makedirs('pretrained', exist_ok=True)
    mge.save(new_dict, os.path.join('pretrained', torch_name + '.pkl'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default='densenet121',
        help=f"which model to convert from torch to megengine, default: densenet121, optional: {list(model_urls.keys())}",
    )
    args = parser.parse_args()
    main(args.model)
