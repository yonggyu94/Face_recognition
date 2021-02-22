from torchvision.datasets import ImageFolder # (img, label) 형태의 데이터셋 구성을 쉽게 할 수 있도록 지원
import torchvision.transforms as T # 이미지 전처리를 지원
from torch.utils.data import DataLoader # 데이터 로더를 쉽게 만들 수 있도록 지원
import torch
import os
import random
from PIL import Image
import glob


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.young_list = glob.glob('/home/nas1_userE/Face_dataset/CASIA_REAL_NATIONAL_AGE/1/**/*')
        self.a = sorted(list(map(str, range(10572))))

    def __getitem__(self, index):
        image_path = self.young_list[index]
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        label = self.a.index(image_path.split('/')[-2])
        return img, label

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.young_list)


def data_loader(root, batch_size, gray_scale, shuffle):
    trans_list = []

    if gray_scale:
        trans_list += [T.Grayscale(num_output_channels=1)]

    trans_list += [
        T.RandomHorizontalFlip(),
        T.ToTensor()]

    if gray_scale:
        trans_list += [T.Normalize(mean=(0.5,), std=(0.5,))]
    else:
        trans_list += [T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]

    transformer = T.Compose(trans_list)
    dataset = ImageFolder(root, transform=transformer)
    dloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
    return dloader, len(dataset.classes), len(dataset)


def young_data_loader(root, batch_size, gray_scale, shuffle):
    trans_list = []

    if gray_scale:
        trans_list += [T.Grayscale(num_output_channels=1)]

    trans_list += [
        T.RandomHorizontalFlip(),
        T.ToTensor()]

    if gray_scale:
        trans_list += [T.Normalize(mean=(0.5,), std=(0.5,))]
    else:
        trans_list += [T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]

    transformer = T.Compose(trans_list)
    dataset = CustomDataset(root, transform=transformer)
    dloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dloader, len(dataset)
