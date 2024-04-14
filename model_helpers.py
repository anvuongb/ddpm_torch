import torch
from typing import List
import os

def save_model(
    filename: str,
    model: torch.nn.Module,
):
    # torch.onnx.export(
    #     model=model,
    #     args=input,
    #     input_names=input_names,
    #     out_names=out_names,
    #     f=filename,
    # )
    s = filename.split("/")[:-1]
    f = os.path.join(*s)
    if not os.path.exists(f):
        os.makedirs(f)

    torch.save(model.state_dict(), filename)


def load_model(filename: str, model: torch.nn.Module) -> torch.nn.Module:
    model = model.load_state_dict(torch.load(filename))
    model = model.eval()
    return model
