#!/usr/bin/env python
# encoding: utf-8
'''
@author: wujiyang
@contact: wujiyang@hust.edu.cn
@file: train.py.py
@time: 2018/12/21 17:37
@desc: train script for deep face recognition
'''

import os
import torch.utils.data
from torch.nn import DataParallel
from datetime import datetime
from backbone.mobilefacenet import MobileFaceNet
from backbone.cbam import CBAMResNet
from backbone.attention import ResidualAttentionNet_56, ResidualAttentionNet_92
from backbone.vggface import VGG16

from margin.ArcMarginProduct import ArcMarginProduct
from margin.MultiMarginProduct import MultiMarginProduct
from margin.CosineMarginProduct import CosineMarginProduct
from margin.SphereMarginProduct import SphereMarginProduct
from margin.MVArcMarginProduct import MVArcMarginProduct
from margin.CenterMarginProduct import CenterMarginProduct
from margin.InnerProduct import InnerProduct
from utils.logging import init_log
from torch.optim import lr_scheduler
import torch.optim as optim
import time
import numpy as np

import argparse
from data import data_loader, young_data_loader
import util

from torch.utils.tensorboard import SummaryWriter


def train(args):
    # gpu init
    multi_gpus = False
    if len(args.gpus.split(',')) > 1:
        multi_gpus = True
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # log init
    save_dir = os.path.join(args.save_dir + args.backbone.upper() + '_' + datetime.now().strftime('%Y%m%d_%H%m%s')
                            + args.exp)
    if os.path.exists(save_dir):
        raise NameError('model dir exists!')
    os.makedirs(save_dir)
    logging = init_log(save_dir)
    _print = logging.info
    writer = SummaryWriter(save_dir)

    # data_loader
    root = os.path.join(args.data_path, args.train_dataset)
    train_loader, cls_num, _ = data_loader(root=root, batch_size=args.batch_size,
                                           gray_scale=args.gray_scale, shuffle=True)

    young_loader, _ = young_data_loader(root=args.young_dataset, batch_size=args.batch_size,
                                           gray_scale=args.gray_scale, shuffle=True)
    young_loader = iter(util.cycle(young_loader))

    # define backbone and margin layer
    if args.backbone == 'MobileFace':
        net = MobileFaceNet()
    elif args.backbone == 'Res50_IR':
        net = CBAMResNet(50, feature_dim=args.feature_dim, mode='ir')
    elif args.backbone == 'SERes50_IR':
        net = CBAMResNet(50, feature_dim=args.feature_dim, mode='ir_se')
    elif args.backbone == 'Res100_IR':
        net = CBAMResNet(100, feature_dim=args.feature_dim, mode='ir')
    elif args.backbone == 'SERes100_IR':
        net = CBAMResNet(100, feature_dim=args.feature_dim, mode='ir_se')
    elif args.backbone == 'Attention_56':
        net = ResidualAttentionNet_56(feature_dim=args.feature_dim)
    elif args.backbone == 'Attention_92':
        net = ResidualAttentionNet_92(feature_dim=args.feature_dim)
    elif args.backbone == "vggface":
        net = VGG16()
    else:
        print(args.backbone, ' is not available!')

    if args.margin_type == 'ArcFace':
        margin = ArcMarginProduct(args.feature_dim, cls_num, s=args.scale_size)
    elif args.margin_type == 'MultiMargin':
        margin = MultiMarginProduct(args.feature_dim, cls_num, s=args.scale_size)
    elif args.margin_type == 'CosFace':
        margin = CosineMarginProduct(args.feature_dim, cls_num, s=args.scale_size)
    elif args.margin_type == 'Softmax':
        margin = InnerProduct(args.feature_dim, cls_num)
    elif args.margin_type == 'SphereFace':
        margin = SphereMarginProduct(args.feature_dim, cls_num)
    elif args.margin_type == 'center':
        margin = CenterMarginProduct(args.feature_dim, cls_num)
    elif args.margin_type == 'MVArc':
        margin = MVArcMarginProduct(args.feature_dim, cls_num)
    else:
        print(args.margin_type, 'is not available!')

    if args.resume:
        print('resume the model parameters from: ', args.net_path, args.margin_path)
        net.load_state_dict(torch.load(args.net_path)['net_state_dict'])
        margin.load_state_dict(torch.load(args.margin_path)['net_state_dict'])

    # define optimizers for different layer
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer_ft = optim.SGD([
        {'params': net.parameters(), 'weight_decay': 5e-4},
        {'params': margin.parameters(), 'weight_decay': 5e-4}
    ], lr=0.1, momentum=0.9, nesterov=True)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[6, 11, 16], gamma=0.1)

    if multi_gpus:
        net = DataParallel(net).to(device)
        margin = DataParallel(margin).to(device)
    else:
        net = net.to(device)
        margin = margin.to(device)


    step = 1
    for epoch in range(1, args.total_epoch + 1):
        exp_lr_scheduler.step()
        # train model
        print('Train Epoch: {}/{} ...'.format(epoch, args.total_epoch))
        net.train()

        since = time.time()
        for i, (img, label) in enumerate(train_loader):
            optimizer_ft.zero_grad()

            img = img[:args.batch_size//2, ].to(device)
            label = label[:args.batch_size//2, ].to(device)

            x, y = next(young_loader)
            if x.size(0) < args.batch_size:
                x, y = next(young_loader)

            x, y = x[:args.batch_size//2, ].to(device), y[:args.batch_size//2, ].to(device)

            img = torch.cat((img, x), dim=0)
            label = torch.cat((label, y), dim=0)

            a = torch.randperm(args.batch_size)

            img = img[a, ]
            label = label[a, ]

            features = net(img)

            output = margin(features, label)
            arc_loss = criterion(output, label)

            arc_loss.backward()
            optimizer_ft.step()

            # print train information
            if step % 100 == 0:
                # current training accuracy
                _, predict = torch.max(output.data, 1)
                total = label.size(0)
                correct = (np.array(predict.cpu()) == np.array(label.data.cpu())).sum()
                time_cur = (time.time() - since) / 100
                since = time.time()
                print("Iters: {:0>6d}/[{:0>2d}], arc_loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}".format(
                            step, epoch, arc_loss.item(), correct / total, time_cur, exp_lr_scheduler.get_lr()[0]))

            # save model
            if step % args.save_freq == 0:
                msg = 'Saving checkpoint: {}'.format(step)
                _print(msg)
                if multi_gpus:
                    net_state_dict = net.module.state_dict()
                    margin_state_dict = margin.module.state_dict()
                else:
                    net_state_dict = net.state_dict()
                    margin_state_dict = margin.state_dict()
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                torch.save({
                    'iters': step,
                    'net_state_dict': net_state_dict},
                    os.path.join(save_dir, 'Iter_%06d_net.ckpt' % step))
                torch.save({
                    'iters': step,
                    'net_state_dict': margin_state_dict},
                    os.path.join(save_dir, 'Iter_%06d_margin.ckpt' % step))

            # test accuracy
            if step % args.test_freq == 0:

                fixed_test_pair = util.fixed_img_list(os.path.join(args.code_path,
                                                                   "txt_files/children.txt"), 1900)
                agedb_acc, th2 = util.verification(net=net, pair_list=fixed_test_pair,
                                               tst_data_dir=os.path.join(args.data_path, args.AgeDB_folder),
                                               gray_scale=args.gray_scale)

                print("AgeDB_Children : %.3f, %.3f" % (agedb_acc, th2))
                writer.add_scalar('acc/agedb_acc', agedb_acc, step)

                fixed_test_pair = util.fixed_img_list(os.path.join(args.code_path,
                                                                   "txt_files/fgnet_children.txt"), 3408)

                fgnet_acc, th2 = util.verification(net=net, pair_list=fixed_test_pair,
                                                   tst_data_dir=os.path.join(args.data_path, args.FGnet_folder),
                                                   gray_scale=args.gray_scale)

                print("FGNet_Children : %.3f, %.3f" % (fgnet_acc, th2))
                writer.add_scalar('acc/fgnet_acc', fgnet_acc, step)

                net.train()
            step += 1
    print('finishing training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    parser.add_argument('--exp', type=str, default="ArcFace_oversampling")
    parser.add_argument('--data_path', type=str, default="/home/nas1_userE/Face_dataset")
    parser.add_argument('--code_path', type=str, default="/home/nas1_userC/yonggyu/Face_Recognition")
    parser.add_argument('--train_dataset', type=str, default='/home/nas1_userE/Face_dataset/CASIA_REAL_NATIONAL')
    parser.add_argument('--young_dataset', type=str, default='/home/nas1_userE/Face_dataset/CASIA_REAL_NATIONAL_Age/0')
    parser.add_argument('--lfw_folder', type=str, default="lfw_single_112_RF")
    parser.add_argument('--AgeDB_folder', type=str, default="AgeDB_single_112_RF")
    parser.add_argument('--FGnet_folder', type=str, default="FGNET_single_112_RF")

    parser.add_argument('--backbone', type=str, default='Res50_IR',
                        help='MobileFace, Res50_IR, SERes50_IR, Res100_IR, SERes100_IR, '
                             'Attention_56, Attention_92, vggface')
    parser.add_argument('--margin_type', type=str, default='ArcFace',
                        help='ArcFace, CosFace, SphereFace, MultiMargin, Softmax, MVArc')

    parser.add_argument('--feature_dim', type=int, default=512, help='feature dimension, 128 or 512 or 2622')
    parser.add_argument('--patch_size', type=int, default=17, help='patch size, 9 or 17 or 33')
    parser.add_argument('--scale_size', type=float, default=32.0, help='scale size')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--total_epoch', type=int, default=30, help='total epochs')

    parser.add_argument('--save_freq', type=int, default=3000, help='save frequency')
    parser.add_argument('--test_freq', type=int, default=3000, help='test frequency')
    parser.add_argument('--resume', type=int, default=False, help='resume model')
    parser.add_argument('--net_path', type=str, default='', help='resume model')
    parser.add_argument('--margin_path', type=str, default='', help='resume model')
    parser.add_argument('--save_dir', type=str, default='./model', help='model save dir')
    parser.add_argument('--gpus', type=str, default='1', help='model prefix')

    parser.add_argument('--gray_scale', action='store_true')

    args = parser.parse_args()
    print(args)
    train(args)
