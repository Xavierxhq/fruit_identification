import os
import sys
from os import path as osp
from pprint import pprint

import numpy as np
import torch
import time

from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import utils.loss as loss

from datasets.data_manager import Tableware
from datasets.data_loader import ImageData
from models import get_baseline_model
from evaluator import Evaluator
from utils.meters import AverageMeter
from utils.loss import TripletLoss
from utils.serialization import Logger
from utils.serialization import save_checkpoint
from utils.transforms import TrainTransform, TestTransform


def load_model(base_model, model_path):
    model_parameter = torch.load(model_path)
    base_model.load_state_dict(model_parameter['state_dict'])
    base_model = base_model.cuda()
    print('model', model_path.split('/')[-1], 'loaded.')
    return base_model


def triplet_example(input1, labels, distmat):
    labels_set = set(labels.numpy())
    label_to_indices = {label: np.where(labels.numpy() == label)[0] for label in labels_set}
    random_state = np.random.RandomState(29)
    input2 = torch.Tensor(input1.size())
    input3 = torch.Tensor(input1.size())

    dis_mp, dis_mn, p_inds, n_inds = loss.hard_example_extream(distmat, labels, True)

    for i, label in enumerate(labels.numpy()):
        input2[i] = input1[int(p_inds[i])]
        input3[i] = input1[int(n_inds[i])]


    """

    for i, label in enumerate(labels.numpy()):
        p_idx = i
        while p_idx == i:
            if len(label_to_indices[label]) == 1:
                break
            p_idx = random_state.choice(label_to_indices[label])
        input2[i] = input1[p_idx]
        n_idx = random_state.choice(label_to_indices[
                random_state.choice(list(labels_set - set([label])))])
        input3[i] = input1[n_idx]
    """

    return input1, input2, input3


def train(model, optimizer, criterion, epoch, print_freq, data_loader):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    start = time.time()
    evaluator = Evaluator(model)
    distmat = evaluator.calDistmat(data_loader)
    is_add_margin = False

    for i, inputs in enumerate(data_loader):
        data_time.update(time.time() - start)

        # model optimizer
        # parse data
        imgs, pids = inputs

        img1, img2, img3 = triplet_example(imgs, pids, distmat)
        # img1, img2, img3 = triplet_example(imgs, pids)
        input1 = img1.cuda()
        input2 = img2.cuda()
        input3 = img3.cuda()

        # forward
        feat1 = model(input1)
        feat2 = model(input2)
        feat3 = model(input3)

        loss = criterion(feat1, feat2, feat3)

        optimizer.zero_grad()
        # backward
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - start)
        losses.update(loss.item())

        start = time.time()

        if (i + 1) % print_freq == 0:
            print('Epoch: [{}][{}/{}]\t'
                  'Batch Time {:.3f} ({:.3f})\t'
                    'Data Time {:.3f} ({:.3f})\t'
                    'Loss {:.6f} ({:.6f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.mean,
                              data_time.val, data_time.mean,
                              losses.val, losses.mean))
        if losses.val <1e-5:
            is_add_margin = True

    param_group = optimizer.param_groups
    print('Epoch: [{}]\tEpoch Time {:.3f} s\tLoss {:.6f}\t'
              'Lr {:.2e}'
              .format(epoch, batch_time.sum, losses.mean, param_group[0]['lr']))
    print()
    return is_add_margin


def trainer(data_pth, a, b, _time=0, layers=50):
    seed = 0

    # dataset options
    height = 128
    width = 128

    # optimization options
    optim = 'Adam'
    max_epoch = 4
    train_batch = 64
    test_batch = 64
    lr = 0.1
    step_size = 40
    gamma = 0.1
    weight_decay = 5e-4
    momentum = 0.9
    test_margin = b
    margin = a
    num_instances = 4
    num_gpu = 1

    # model options
    last_stride = 1
    pretrained_model_18 = 'model/resnet18-5c106cde.pth'
    pretrained_model_50 = 'model/resnet50-19c8e357.pth'
    pretrained_model_34 = 'model/resnet34-333f7ec4.pth'
    pretrained_model_101 = 'model/resnet101-5d3b4d8f.pth'
    pretrained_model_152 = 'model/resnet152-b121ed2d.pth'

    # miscs
    print_freq = 20
    eval_step = 1
    save_dir = 'model/pytorch-ckpt/time%d' % _time
    workers = 1
    start_epoch = 0


    torch.manual_seed(seed)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print('currently using GPU')
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(seed)
    else:
        print('currently using cpu')

    pin_memory = True if use_gpu else False

    print('initializing dataset {}'.format('Tableware'))
    dataset = Tableware(data_pth)

    trainloader = DataLoader(
        ImageData(dataset.train, TrainTransform(height, width)),
        batch_size=train_batch, num_workers=workers,
        pin_memory=pin_memory, drop_last=True
    )

    # testloader = DataLoader(
    #     ImageData(dataset.test, TestTransform(height, width)),
    #     batch_size=test_batch, num_workers=workers,
    #     pin_memory=pin_memory, drop_last=True
    # )

    # model, optim_policy = get_baseline_model(model_path=pretrained_model)
    if layers == 18:
        model, optim_policy = get_baseline_model(model_path=pretrained_model_18, layers=18)
    else:
        model, optim_policy = get_baseline_model(model_path=pretrained_model_50, layers=50)
    # model, optim_policy = get_baseline_model(model_path=pretrained_model_18, layers=18)
    # model, optim_policy = get_baseline_model(model_path=pretrained_model_34, layers=34)
    # model, optim_policy = get_baseline_model(model_path=pretrained_model_101, layers=101)
    # model = load_model(model, model_path='./model/pytorch-ckpt/87_layers18_margin20_epoch87.tar')
    print('model\'s parameters size: {:.5f} M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    inner_dist = 0
    outer_dist = 0
    max_outer = 0
    min_outer = 0
    max_iner = 0
    min_iner = 0

    tri_criterion = TripletLoss(margin)

    # get optimizer
    optimizer = torch.optim.Adam(
        optim_policy, lr=lr, weight_decay=weight_decay
    )

    def adjust_lr(optimizer, ep):
        if ep < 20:
            lr = 1e-4 * (ep + 1) / 2
        elif ep < 80:
            lr = 1e-3 * num_gpu
        elif ep < 180:
            lr = 1e-4 * num_gpu
        elif ep < 300:
            lr = 1e-5 * num_gpu
        elif ep < 320:
            lr = 1e-5 * 0.1 ** ((ep - 320) / 80) *num_gpu
        elif ep < 400:
            lr = 1e-6
        elif ep < 480:
            lr = 1e-4 * num_gpu
        else:
            lr = 1e-5 * num_gpu
        for p in optimizer.param_groups:
            p['lr'] = lr

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    evaluator = Evaluator(model)

    for epoch in range(start_epoch, max_epoch):
        if step_size > 0:
            adjust_lr(optimizer, epoch + 1)
        next_margin = margin


        # skip if not save model
        if eval_step > 0 and (epoch + 1) % eval_step == 0 or (epoch + 1) == max_epoch:
            save_record_path = 'margin_'+ str(margin) + '_epoch_' + str(epoch + 1) + '.txt'
            _t1 =time.time()
            train(model, optimizer, tri_criterion, epoch, print_freq, trainloader)
            _t2 = time.time()
            print('time for training:', '%.2f' % (_t2 - _t1), 's')

            """
            acc, inner_dist, outer_dist, max_outer, min_outer, max_iner, min_iner = evaluator.evaluate(testloader, test_margin, save_record_path)
            print('margin:{}, epoch:{}, acc:{}'.format(margin, epoch+1, acc))
            f = open('record.txt', 'a')
            f.write('margin:{}, epoch:{}, acc:{}\n'.format(margin, epoch+1, acc))
            f.close()
            """

            is_best = False
            # save_model_path = 'new_margin({})_epoch({}).pth.tar'.format(margin, epoch+1)
            save_model_path = 'time{}_layers{}_margin{}_epoch{}.tar'.format(_time, layers, margin, epoch+1)
            # save_model_path = 'layers34_margin{}_epoch{}.tar'.format(margin, epoch+1)
            # save_model_path = 'layers101_margin{}_epoch{}.tar'.format(margin, epoch+1)
            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()


            save_checkpoint({
                'state_dict': state_dict,
                'epoch': epoch + 1,
            }, is_best=is_best, save_dir=save_dir, filename=save_model_path)
            # in this place, is_best is always True.
            # so it will think that the newest model is also the best one.

    # print(
    #     'Best accuracy {:.1%}, achieved at epoch {}'.format(best_acc, best_epoch))
    # f.close()

    #         do_get_feature_and_t(os.path.join('model/pytorch-ckpt', '1_' + save_model_path), margin=margin, epoch=epoch+1)
            margin = next_margin
    return save_model_path, inner_dist, outer_dist, max_outer, min_outer, max_iner, min_iner


if __name__ == "__main__":

    # for margin in range(1, 500):
    #     f = open('margin_dist.txt', 'a')
    #     inner_dist, outer_dist, max_outer, min_outer, max_iner, min_iner = trainer('/home/ubuntu/Program/Tableware/reid_tableware/datas/dishes_dataset/', margin, 0)
    #     f.write("{},{},{},{},{},{},{}\r\n".format(margin,inner_dist,outer_dist, max_outer, min_outer, max_iner, min_iner))
    #     print(margin, inner_dist, outer_dist)
    #     f.close()

    for _i in range(3):
        trainer('./train_data/', 20, 0, _time=_i+1, layers=18)
        trainer('./train_data/', 20, 0, _time=_i+1, layers=50)
    # _ = don't care.

    # model_path = '1_margin(10)_epoch(1).pth.tar'


    # it will cost you lots of time.
