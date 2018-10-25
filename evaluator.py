from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import torch

from utils import loss

class Evaluator(object):
    def __init__(self, model):
        self.model = model

    def calDistmat(self, data_loader):
        for i, inputs in enumerate(data_loader):
            imgs, pids = inputs
            data = imgs.cuda()
            with torch.no_grad():
                feature = self.model(data)
            feature.cpu()
            distmat = loss.euclidean_dist(feature, feature).cpu()
        return distmat

    def evaluate(self, data_loader, margin, save_file_name):
        # f = open(save_file_name, 'w+')
        count = 1
        summ = 0.0

        inner_num = 1
        inner_dist = 0
        outer_num = 1
        outer_dist = 0
        max_dist_inner = 0
        max_dist_outer = 0
        min_dist_outer = 1000000
        min_dist_inner = 1000000
        for i, inputs in enumerate(data_loader):
            imgs, pids = inputs
            data = imgs.cuda()
            with torch.no_grad():
                feature = self.model(data)
            feature.cpu()
            _inner_list = []
            _outer_list = []
            distmat = loss.euclidean_dist(feature, feature).cpu()
            for i, label_i in enumerate(pids):
                for j, label_j in enumerate(pids):

                    if i == j:
                        continue
                        # f.write('{}, {} :{}\r\n'.format(label_i, label_j, distmat[i, j]))
                    if label_i == label_j:
                        # inner_num += 1
                        # inner_dist += distmat[i, j]
                        _inner_list.append(distmat[i, j])
                        # max_dist_inner = max(max_dist_inner, distmat[i, j])
                        # min_dist_inner = min(min_dist_inner, distmat[i, j])
                    else:
                        # outer_num += 1
                        # outer_dist += distmat[i, j]
                        _outer_list.append(distmat[i, j])
                        # max_dist_outer = max(max_dist_outer, distmat[i, j])
                        # min_dist_outer = min(min_dist_outer, distmat[i, j])


            for i in _inner_list:
                for j in _outer_list:
                    if i < j:
                        summ += 1
                    count += 1


        return summ / count, inner_dist/inner_num, outer_dist/outer_num, max_dist_outer, min_dist_outer, max_dist_inner, min_dist_inner
