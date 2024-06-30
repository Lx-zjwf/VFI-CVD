import os
import numpy as np
import shutil

data_path = '/home/luxin/Few-shot Fine-grained/meta_iNat/train'
tgt_path = '/home/luxin/Few-shot Fine-grained/iNat_close'
if not (os.path.exists(tgt_path)):
    os.mkdir(tgt_path)
train_ratio = 0.7
class_list = os.listdir(data_path)
class_path = [os.path.join(data_path, cls) for cls in class_list]
for path in class_path:
    cls = path.split('/')[-1] + '/'
    image_list = os.listdir(path)
    train_num = int(len(image_list) * train_ratio)
    train_list = image_list[:train_num]
    if train_num < 20:
        train_list = np.random.choice(train_list, 20).tolist()
    val_list = image_list[train_num:]
    for i in range(len(train_list)):
        image = train_list[i]
        image_path = os.path.join(path, image)
        train_path = os.path.join(tgt_path, 'train', cls)
        if not os.path.exists(train_path):
            os.mkdir(train_path)
        shutil.copy(image_path, train_path)
    for i in range(len(val_list)):
        image = val_list[i]
        image_path = os.path.join(path, image)
        val_path = os.path.join(tgt_path, 'val', cls)
        if not os.path.exists(val_path):
            os.mkdir(val_path)
        test_path = os.path.join(tgt_path, 'test', cls)
        if not os.path.exists(test_path):
            os.mkdir(test_path)
        shutil.copy(image_path, val_path)
        shutil.copy(image_path, test_path)
