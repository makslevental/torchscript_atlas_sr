from collections import OrderedDict
from pprint import pprint

import torch
from torch import nn

from dbpn import DBPNITER
from edsr import EDSR

torch.manual_seed(1)


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


# model = Generator(4)
upscale_factor = 2
# model = DBPNITER(
#     num_channels=3,
#     base_filter=64,
#     feat=256,
#     num_stages=3,
#     scale_factor=upscale_factor,
# )

n_resblocks = 32
n_feats = 256
model = EDSR(upscale_factor, n_resblocks, n_feats, res_scale=0.1)

model = load_model_state(
    model, "/home/maksim/data/checkpoints/dbpn_checkpoints/edsr_x2.pt"
)
# An example input you would normally provide to your model's forward() method.
# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.script(model)
output = traced_script_module(torch.ones(1, 3, 100, 100))
pprint(output[0, 1, 1, :10].detach().numpy().tolist())

traced_script_module.save("traced_dbpn_model.pt")
