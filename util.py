import torch
import random
import torchvision.transforms as T
from PIL import Image
import os
import numpy as np


def make_dir(args):
    model_path = os.path.join(args.exp, args.model_dir)
    log_path = os.path.join(args.exp, args.log_dir)

    if not os.path.isdir(args.exp):
        os.mkdir(args.exp)

    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    if not os.path.isdir(log_path):
        os.mkdir(log_path)

    return model_path, log_path


def cos_dist(x1, x2):
    return torch.sum(x1 * x2) / (torch.norm(x1) * torch.norm(x2))


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def fixed_img_list(lfw_pair_text, test_num):

    f = open(lfw_pair_text, 'r')
    lines = []

    while True:
        line = f.readline()
        if not line:
            break
        lines.append(line)
    f.close()

    random.shuffle(lines)
    lines = lines[:test_num]
    return lines


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def verification(net, pair_list, tst_data_dir, gray_scale=True):
    similarities = []
    labels = []

    # 이미지 전처리
    trans_list = []

    if gray_scale:
        trans_list += [T.Grayscale(num_output_channels=1)]

    trans_list += [
                   T.ToTensor()
    ]

    if gray_scale:
        trans_list += [T.Normalize(mean=(0.5,), std=(0.5,))]
    else:
        trans_list += [T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    t = T.Compose(trans_list)

    # 주어진 모든 이미지 pair에 대해 similarity 계산
    net.eval()
    with torch.no_grad():  # Test 때 GPU를 사용할 경우 메모리 절약을 위해 torch.no_grad() 내에서 하는 것이 좋다.
        for idx, pair in enumerate(pair_list):
            # Read paired images
            path_1, path_2, label = pair.split(' ')
            img_1 = t(Image.open(os.path.join(tst_data_dir, path_1))).unsqueeze(dim=0).cuda()
            img_2 = t(Image.open(os.path.join(tst_data_dir, path_2))).unsqueeze(dim=0).cuda()
            imgs = torch.cat((img_1, img_2), dim=0)

            # Extract feature and save
            features = net(imgs).cpu()
            similarities.append(cos_dist(features[0], features[1]))
            labels.append(int(label))

    '''
    STEP 2 : similarity와 label로 verification accuracy 측정
    '''
    best_accr = 0.0
    best_th = 0.0

    # 각 similarity들이 threshold의 후보가 된다
    list_th = similarities

    # list -> tensor
    similarities = torch.stack(similarities, dim=0)
    labels = torch.ByteTensor(labels)

    # 각 threshold 후보에 대해 best accuracy를 측정
    for i, th in enumerate(list_th):
        pred = (similarities >= th)
        correct = (pred == labels)
        accr = torch.sum(correct).item() / correct.size(0)

        if accr > best_accr:
            best_accr = accr
            best_th = th.item()

    return best_accr, best_th
