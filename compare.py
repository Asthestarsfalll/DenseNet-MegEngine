import time

import megengine as mge
import numpy as np
import torch

from models.densenet import densenet121, densenet161, densenet169, densenet201
from models.torch_model import densenet121 as torch_densenet121
from models.torch_model import densenet161 as torch_densenet161
from models.torch_model import densenet169 as torch_densenet169
from models.torch_model import densenet201 as torch_densenet201

mge_model = densenet121(True)
# mge_model = densenet161(True)
# mge_model = densenet169(True)
# mge_model = densenet201(True)

# download manually if speed is too slow
torch_model = torch_densenet121(pretrained=True)
# torch_model = torch_densenet161(pretrained=True)
# torch_model = torch_densenet169(pretrained=True)
# torch_model = torch_densenet201(pretrained=True)

# mge_model.load_state_dict(mge.load('./pretrained/densenet161.pkl'))

mge_model.eval()
torch_model.eval()

torch_time = meg_time = 0.0

def test_func(mge_out, torch_out):
    result = np.isclose(mge_out, torch_out, rtol=1e-3)
    ratio = np.mean(result)
    allclose = np.all(result) > 0
    abs_err = np.mean(np.abs(mge_out - torch_out))
    std_err = np.std(np.abs(mge_out - torch_out))
    return ratio, allclose, abs_err, std_err


def softmax(logits):
    logits = logits - logits.max(-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(-1, keepdims=True)


for i in range(15):
    results = []
    inp = np.random.randn(2, 3, 224, 224)
    mge_inp = mge.tensor(inp, dtype=np.float32)
    torch_inp = torch.tensor(inp, dtype=torch.float32)

    if torch.cuda.is_available():
        torch_inp = torch_inp.cuda()
        torch_model.cuda()

    st = time.time()
    mge_out = mge_model(mge_inp)
    meg_time += time.time() - st

    st = time.time()
    torch_out = torch_model(torch_inp)
    torch_time += time.time() - st

    if torch.cuda.is_available():
        torch_out = torch_out.detach().cpu().numpy()
    else:
        torch_out = torch_out.detach().numpy()
    mge_out = mge_out.numpy()
    mge_out = softmax(mge_out)
    torch_out = softmax(torch_out)
    ratio, allclose, abs_err, std_err = test_func(mge_out, torch_out)
    results.append(allclose)
    print(f"Result: {allclose}, {ratio*100 : .4f}% elements is close enough\n which absolute error is  {abs_err} and absolute std is {std_err}")

assert all(results), "not aligned"

print(f"meg time: {meg_time}, torch time: {torch_time}")
