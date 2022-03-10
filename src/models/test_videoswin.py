import torch
# import torch.nn as nn
# from collections import OrderedDict
# from VideoSwin import SwinTransformer3D

from einops import repeat


y = torch.randint(1, 10, (1, 1, 2, 4, 4))
print(y)
y = torch.repeat_interleave(y, 4, dim=2)
# y = repeat(y, 'bcfhw -> bc(f repeat)hw', repeat=4)
print(y)