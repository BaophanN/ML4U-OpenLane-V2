from mmdet.models import FPN 
import torch 
"""
import mmdet 
print(mmdet.__file__)
>>/opt/conda/envs/lanesegnet/lib/python3.8/site-packages/mmdet/__init__.py
"""

in_channels = [128,256,512]
_num_levels_ = 4
_dim_ = 256
import torch

in_channels = [2, 3, 5, 7]
scales = [340, 170, 84, 43]
inputs = [torch.rand(1, c, s, s)
          for c, s in zip(in_channels, scales)]
# self = FPN(in_channels, 11, len(in_channels)).eval()
# outputs = self.forward(inputs)
# for i in range(len(outputs)):
#     print(f'outputs[{i}].shape = {outputs[i].shape}')
for c,s in zip(in_channels,scales):
    print(c,s)
"""
1 2 340 340 
1 3 170 170 
1 5 84  84 
1 7 43  43 
"""