# DenseNet-MegEngine

The MegEngine Implementation of DenseNet(Densely Connected Convolutional Networks).

## Usage

Install dependency.

```bash
pip install -r requirements.txt
```

If you don't want to compare the ouput error between the MegEngine implementation and PyTorch one,  just ignore requirements.txt and install MegEngine from the command line:

```bash
python3 -m pip install --upgrade pip 
python3 -m pip install megengine -f https://megengine.org.cn/whl/mge.html
```

Convert trained weights from torch to megengine, the converted weights will be saved in ./pretained/

```bash
python convert_weights.py -m densenet121
```

Import from megengine.hub:

Way 1:

```python
from  megengine import hub
modelhub = hub.import_module(repo_info='asthestarsfalll/densenet-megengine', git_host='github.com')

# load DenseNet model and custom on you own
resnest = modelhub.DenseNet(32, (6, 12, 24, 16), 64, num_classes=10)

# load pretrained model 
pretrained_model = modelhub.densenet121(pretrained=True) 
```

Way 2:

```python
from  megengine import hub

# load pretrained model 
model_name = 'densenet121'
pretrained_model = hub.load(
    repo_info='asthestarsfalll/densenet-megengine', entry=model_name, git_host='github.com', pretrained=True)
```

Currently support densenet121 and densenet161,  but you can run convert_weights.py to convert other models(densenet169 and densenet201).
For example:

```bash
  python convert_weights.py -m densenet201
```

Then load state dict manually.

```python
model = modelhub.densenet201()
model.load_state_dict(mge.load('./pretrained/densenet201.pkl'))s
# or
model_name = 'densenet101'
model = hub.load(
    repo_info='asthestarsfalll/densenet-megengine', entry=model_name, git_host='github.com')
model.load_state_dict(mge.load('./pretrained/densenet201.pkl'))
```

## TODO

- [ ] add train/test codes maybe
- [ ] add some introducations about DenseNet and the way to implement it(maybe)

## Reference

[The PyTorch implementation of DenseNet](https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py)
