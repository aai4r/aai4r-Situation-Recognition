import pickle
import os
import json
import time
import numpy as np
from collections import Counter
import copy

import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


from PIL import Image
import matplotlib.pyplot as plt

from utils import save_checkpoint, AverageMeter
from model import GNN_batch
from dataset import DriveData, DriveData_test
from dataset_utils import *
from train import train_model
from validate import validate_model


max_num_nodes = 6

train = json.load(open("Dataset/imsitu/train_v1.json"))
test = json.load(open("Dataset/imsitu/test_v1.json"))


params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 1}

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

dset_train = DriveData(train, transforms.Compose([
    transforms.Resize(224),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
]))


train_loader = DataLoader(dset_train, **params)

params = {'batch_size': 1,
          'shuffle': False,
          'num_workers': 1}

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

dset_test = DriveData_test(test, transforms.Compose([
    transforms.Resize(224),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
]))

test_loader = DataLoader(dset_test, **params)


print("Loaded the dataset")

backbone = 'resnet101_v4.1'
if backbone == 'resnet101_v4.1':
    from baseline_crf import BaselineCRF

    model = BaselineCRF(encoding=torch.load(
        'output_crf_v1/encoder'), cnn_type='resnet_101')
    model.load_state_dict(torch.load('output_crf_v1/best.model'))
    feature_length = model.cnn.rep_size()

    class ResNet101_v4_1(nn.Module):
        def __init__(self, crf):
            super(ResNet101_v4_1, self).__init__()
            self.crf = crf
            self.perm = torch.tensor([6, 7, 2, 3, 8, 10, 1, 0, 4, 9, 5, 11])

        def forward(self, image):
            return self.crf.forward_max(image)[0][:, self.perm]

    # verb_model = ResNet101_v4(model)
    verb_model = ResNet101_v4_1(model)
    verb_feature_model = model.cnn
    noun_model = model.cnn

verb_model.eval()
verb_feature_model.eval()
verb_model = verb_model.cuda()
verb_feature_model = verb_feature_model.cuda()
for param in verb_model.parameters():
    param.requires_grad = False
print("Load the verb model")

noun_model.eval()
noun_model = noun_model.cuda()
for param in noun_model.parameters():
    param.requires_grad = False
print("load the noun model")

gnn = GNN_batch(noun_vocabulary_size, verb_vocabulary_size,
                role_vocabulary_size, hidden_dimension, feature_length, 1)

if use_cuda:
    criterion = nn.CrossEntropyLoss(reduce=False).cuda()
    gnn = gnn.cuda()
else:
    criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.RMSprop(gnn.parameters(),
                                lr=learning_rate)
save_postfix = 'crf_rmsprop_lr{:.0e}'.format(learning_rate)
is_best = 0

for epoch in range(start_epoch, 0):

    if epoch > 10:

        print("Decay learning rate \n")
        learning_rate = .85*learning_rate

        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

    train_model(train_loader, gnn, verb_feature_model,
                noun_model, criterion, optimizer, epoch)

    # gnn.load_state_dict(torch.load('model_best.pth.tar')['state_dict'])
    top1_verb, top1_value, top1_value_all, top5_verb, top5_value, top5_value_all = validate_model(test_loader, gnn, verb_model,
                                                                                                  verb_feature_model, noun_model, criterion)
    _, gt_value, gt_value_all, _, _, _ = validate_model(test_loader, gnn, None,
                                                        verb_feature_model, noun_model, criterion)
    mean_score = sum([top1_verb, top1_value, top1_value_all,
                      top5_verb, top5_value, top5_value_all,
                      gt_value, gt_value_all]) / 8
    print('top1-verb \t top1-value \t top1-value-all \t top5-verb \t top5-value \t top5-value-all \t gt-value \t gt-value-all \n'
          '{0:.4f} \t {1:.4f} \t {2:.4f} \t {3:.4f} \t {4:.4f} \t {5:.4f} \t {6:.4f} \t {7:.4f}'.format(
              top1_verb, top1_value, top1_value_all, top5_verb, top5_value, top5_value_all, gt_value, gt_value_all))
    print('mean score: {0:.4f}'.format(mean_score))

    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': gnn.state_dict()
    },
        mean_score > is_best,
        filename='models/{}_{}_T_4/gnn_model_three_anotation_test'.format(backbone, save_postfix) + str(epoch) + 'pth.tar')
    if mean_score > is_best:
        is_best = mean_score
    save_checkpoint({
        'epoch': epoch+1,
        'state_dict': verb_model.state_dict(),
    },
        False,
        filename='models/{}_{}_T_4/verb_model_three_anotation_test'.format(backbone, save_postfix) + str(epoch) + 'pth.tar')
    save_checkpoint({
        'epoch': epoch+1,
        'state_dict': noun_model.state_dict(),
    },
        False,
        filename='models/{}_{}_T_4/noun_model_three_anotation_test'.format(backbone, save_postfix) + str(epoch) + 'pth.tar')
gnn.load_state_dict(torch.load('model_best.pth.tar')['state_dict'])
top1_verb, top1_value, top1_value_all, top5_verb, top5_value, top5_value_all = \
    validate_model(test_loader, gnn, verb_model,
                   verb_feature_model, noun_model, criterion)
_, gt_value, gt_value_all, _, _, _ = \
    validate_model(test_loader, gnn, None,
                   verb_feature_model, noun_model, criterion)
mean_score = sum([top1_verb, top1_value, top1_value_all,
                  top5_verb, top5_value, top5_value_all,
                  gt_value, gt_value_all]) / 8
print('top1-verb \t top1-value \t top1-value-all \t top5-verb \t top5-value \t top5-value-all \t gt-value \t gt-value-all \n'
      '{0:.4f} \t {1:.4f} \t {2:.4f} \t {3:.4f} \t {4:.4f} \t {5:.4f} \t {6:.4f} \t {7:.4f}'.format(
          top1_verb, top1_value, top1_value_all, top5_verb, top5_value, top5_value_all, gt_value, gt_value_all))
print('mean score: {0:.4f}'.format(mean_score))
