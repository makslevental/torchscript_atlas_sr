import argparse
from collections import OrderedDict
from os.path import basename, splitext
from pprint import pprint

import torch
from torch import nn

from dbpn import DBPNITER
from edsr import EDSR
from srresnet import Generator


def remove_module_load(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    return new_state_dict


def load_model_state(model: nn.Module, fp: str, remove_module=True, strict=True):
    pretrained_dict = torch.load(fp, map_location="cpu")
    if remove_module:
        pretrained_dict = remove_module_load(pretrained_dict)
    try:
        if strict:
            model.load_state_dict(pretrained_dict)
        else:
            model_dict = model.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() if k in model_dict
            }
            model_dict.update(pretrained_dict)
    except:
        print("\npretrained\n")
        pprint(list(pretrained_dict.keys()))
        print("\nmodel\n")
        pprint(list(model.state_dict().keys()))
        raise
    del pretrained_dict
    return model


parser = argparse.ArgumentParser()
parser.add_argument("--model")
parser.add_argument("--upscale", type=int)

args = parser.parse_args()

upscale_factor = args.upscale
model_name = args.model
model_fp = f"{model_name}_{upscale_factor}x.pth"

# srresnet
if model_name == "srresnet":
    model = Generator(upscale_factor)
elif model_name == "dbpn":
    model = DBPNITER(
        num_channels=3,
        base_filter=64,
        feat=256,
        num_stages=3,
        scale_factor=upscale_factor,
    )
elif model_name == "edsr":
    model = EDSR(upscale_factor, n_resblocks=32, n_feats=256, res_scale=0.1)
else:
    raise Exception("unknown model name")

model = load_model_state(
    model, model_fp
)
traced_script_module = torch.jit.script(model)

bn = splitext(basename(model_fp))[0]

traced_script_module.save(f"traced_{bn}.pt")
