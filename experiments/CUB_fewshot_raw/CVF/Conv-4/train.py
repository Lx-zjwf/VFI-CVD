import os
import sys
import torch
import yaml
from functools import partial
import clip

sys.path.append('../../../../')
from trainers import trainer, cvfi_train
from datasets import dataloaders
from models.VD_CVFI import VD_CVFI

args = trainer.train_parser()

with open('../../../../config.yml', 'r') as f:
    temp = yaml.safe_load(f)
data_path = os.path.abspath(temp['data_path'])
fewshot_path = os.path.join(data_path, 'CUB_fewshot_raw')

pm = trainer.Path_Manager(fewshot_path=fewshot_path, args=args)
train_way = args.train_way
shots = [args.train_shot, args.train_query_shot]
train_loader = dataloaders.meta_train_dataloader(data_path=pm.train,
                                                 way=train_way,
                                                 shots=shots,
                                                 transform_type=args.train_transform_type)

model = VD_CVFI(way=train_way, shots=[args.train_shot, args.train_query_shot])
# 'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14'
vlm, preprocess = clip.load('ViT-B/32', device='cuda')

train_func = partial(cvfi_train.default_train, train_loader=train_loader)
tm = trainer.Train_Manager(args, path_manager=pm, train_func=train_func)
tm.train(model, vlm, preprocess)
tm.evaluate(model, vlm, preprocess)
