import sys
import os
import torch
import yaml
import clip

sys.path.append('../../../../')
from models.VD_CVFI import VD_CVFI
from utils import util
from trainers.eval import meta_test

with open('../../../../config.yml', 'r') as f:
    temp = yaml.safe_load(f)
data_path = os.path.abspath(temp['data_path'])

test_path = os.path.join(data_path, 'dogs/test')
model_path = './model_Clip-FRN.pth'

gpu = 1
torch.cuda.set_device(gpu)

model = VD_CVFI()
model.cuda()
model.load_state_dict(torch.load(model_path, map_location=util.get_device_map(gpu)), strict=True)
model.eval()
vlm, preprocess = clip.load('ViT-B/32', device='cuda')

with torch.no_grad():
    way = 5
    for shot in [1, 5]:
        mean, interval = meta_test(data_path=test_path,
                                   model=model,
                                   vlm=vlm,
                                   preprocess=preprocess,
                                   way=way,
                                   shot=shot,
                                   pre=None,
                                   transform_type=0,
                                   trial=2000)
        print('%d-way-%d-shot acc: %.3f\t%.3f' % (way, shot, mean, interval))
