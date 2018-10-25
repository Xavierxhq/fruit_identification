# encoding: utf-8
"""
@author:  liaoxingyu
@contact: xyliao1993@qq.com 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)
    print(N)
    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


def hard_example_extream(dist_mat, labels, return_inds=True):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    np.set_printoptions(threshold=1e6)
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)
    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
    temp_p_dist_mat = torch.Tensor(dist_mat.size())
    temp_n_dist_mat = torch.Tensor(dist_mat.size())
    temp_p_inds = labels.expand(N, N).eq(labels.expand(N, N).t())
    temp_n_inds = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    for i in range(N):
        for j in range(N):
            if is_pos[i][j] == 1:
                temp_p_dist_mat[i][j] = dist_mat[i][j]
            else:
                temp_p_dist_mat[i][j] = 0

    for i in range(N):
        for j in range(N):
            if is_neg[i][j] == 1:
                temp_n_dist_mat[i][j] = dist_mat[i][j]
            else:
                temp_n_dist_mat[i][j] = 127

    dist_ap, relative_p_inds = torch.max(
        temp_p_dist_mat.contiguous().view(N, -1), 1, keepdim=True)

    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        temp_n_dist_mat.contiguous().view(N, -1), 1, keepdim=True)

    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
            .copy_(torch.arange(0, N).long())
            .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        for i in range(N):
            for j in range(N):
                if is_pos[i][j] == 1:
                    temp_p_inds[i][j] = ind[i][j]
                else:
                    temp_p_inds[i][j] = 0

        for i in range(N):
            for j in range(N):
                if is_neg[i][j] == 1:
                    temp_n_inds[i][j] = ind[i][j]
                else:
                    temp_n_inds[i][j] = 0

        p_inds = torch.gather(
            temp_p_inds.contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            temp_n_inds.contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        relative_p_inds = relative_p_inds.squeeze(1)
        relative_n_inds = relative_n_inds.squeeze(1)

        """
        for i in range(N):
            print(relative_p_inds.size())
            print(labels[relative_p_inds[i]])
            print("&&")
            for j in range(N):
                if(is_pos[i][j] == 1):
                    print(labels[i])
                    print(labels[j])
            print("--------------------")
        print("fuck")

        """

        return dist_ap, dist_an, relative_p_inds, relative_n_inds

    return dist_ap, dist_an
    

class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None):
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)

    def __call__(self, anchor, positive, nagative, normalize_feature=False):
        if normalize_feature:
            anchor = normalize(anchor, axis=-1)
            positive = normalize(positive, axis=-1)
            nagative = normalize(nagative, axis=-1)
        loss = self.triplet_loss(anchor, positive, nagative)
        return loss

class triplet_my_loss(nn.Module):
    def __init__(self, margin=1.,use_gpu=True):
        super(triplet_my_loss, self).__init__()
        self.use_gpu = use_gpu
        self.margin = margin
        self.mse = nn.MSELoss()
        # self.mse = nn.PairwiseDistance(2)
        #.inputs = inputs

    def forward(self, inputs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """

        p = inputs[0]
        p1 = inputs[1]
        n1 = inputs[2]
        anchor = normalize(p, axis=-1)
        positive = normalize(p1, axis=-1)
        negative = normalize(n1, axis=-1)
        s1 = torch.sum(self.mse(anchor, positive))
        s2 = torch.sum(self.mse(anchor, negative))
        #
        # print(s1)
        # print(s2)
        loss = torch.mul(torch.mul(s1, self.margin), torch.pow(s2, -1))
        # print(loss)
        return loss



class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss
