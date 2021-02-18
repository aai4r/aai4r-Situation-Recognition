import pickle
import os
import json
import time
import numpy as np
from collections import Counter
import copy
import torchvision

import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2
from torchvision import datasets
from PIL import Image

from utils import save_checkpoint, AverageMeter
from model import GNN_batch
# from dataset import DriveData, DriveData_test
from dataset_utils import *
# from train import train_model
# from validate import validate_model

import time
import torch.nn as nn
import torch
import torch.nn.functional as F
from dataset_utils import *
from PIL import Image

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
    gnn = gnn.cuda()
gnn.load_state_dict(torch.load('model_best_online.pth.tar')['state_dict'])
gnn.eval()

# create index_to_noun for predict noun by index
indextonoun = {v:k for k,v in noun2index.items()}

# set cam
CAM_ID=0
cap = cv2.VideoCapture(CAM_ID)
cv2.namedWindow("predicted output", cv2.WINDOW_NORMAL)

if cap.isOpened() ==False:
    print ('cant open the CAM(%d)' %(CAM_ID))
    exit()

# for video input
while True:
    ret, frame = cap.read()
    if ret :
        # image preprocessing
        frame_cp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_cp = cv2.resize(frame_cp, dsize=(224, 224), interpolation=cv2.INTER_AREA)
        frame_cp = frame_cp[:, :, ::-1].transpose((2, 0, 1)).copy()
        # convert to Tensor
        frame_cp = torch.from_numpy(frame_cp).float().div(255.0)
        # normalizing
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        frame_cp = norm(frame_cp)
        frame_cp = frame_cp.unsqueeze(0)

        input_image = frame_cp
        input_var = torch.autograd.Variable(input_image).cuda()

        verb_features = verb_feature_model(input_var)
        noun_features = noun_model(input_var)

        # transform noun and verb_features to hidden representation
        hidden_verb_features = gnn.weight_iv(verb_features)
        hidden_noun_features = gnn.weight_in(noun_features).unsqueeze(1)

        image_verbs = verb_model(input_var)
        verb_prob = torch.max(torch.nn.functional.softmax(image_verbs, dim=1)).item() * 100

        image_verbs = image_verbs.argsort(1, descending=True).cpu()
        image_verbs = image_verbs[0, :1]
        image_verbs = image_verbs.data.cpu().numpy()
        image_num = image_verbs[0]
        print('predicted verb is : ', verb_vocabulary[image_num], verb_prob )

        image_frame = [verb2roles_with_ids[x] for x in image_verbs]
        image_roles = [[role2index[x] for x in frame] for frame in image_frame]
        image_roles = np.array([i + [0] * (max_num_nodes - len(i)) for i in image_roles])

        role_graphs = []
        for verb in image_verbs:
            role_graphs.append(verb2role_graph[verb])

        role_adj_matrix, role_mask = gnn.build_adj_matrix(role_graphs)

        init_state = np.zeros((batch_size, max_num_nodes, hidden_dimension))

        W_v = gnn.verb_embedding(torch.from_numpy(image_verbs).cuda()).unsqueeze(1)
        W_e = gnn.role_embedding(torch.from_numpy(image_roles).cuda())

        init_states = gnn.inside_active(hidden_noun_features * W_e * W_v)

        # forward pass
        output = gnn(init_states, role_adj_matrix, role_mask, n_steps=T)
        output = gnn.classifier(output)

        output = output.squeeze(0)
        output_label = output.argsort(1, descending=True)

        # to print probability of nouns (total 6 nouns exist)
        output_s = [0] * 6
        noun_prob = [0] * 6
        for i in range(6):
            output_s[i] = output[i:i + 1]
            noun_prob[i] = torch.max(torch.nn.functional.softmax(output[i], dim=0)).item() * 100

        output1 = output_label[:, :1].squeeze(1)
        num_output = output1.tolist()

        # print verb prediction
        verbtext = 'predicted verb is : {c1} / prob : {c2}'.format(c1=verb_vocabulary[image_num],c2=round(verb_prob,2))
        if noun_prob[i] > 95.0:
            cv2.putText(frame, verbtext, (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255))
        else:
            cv2.putText(frame, verbtext, (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0))

        cnt = 0
        for i in range(len(image_frame[0])):
            if num_output[i] != 0:
                roletext = 'role : {c1} / noun : {c2} / prob : {c3}'.format(c1=image_frame[0][i], c2=indextonoun[num_output[i]],c3=round(noun_prob[i],2))
                print('role :', image_frame[0][i], '/ noun :', indextonoun[num_output[i]], '/ prob :', noun_prob[i])
                if noun_prob[i] > 65.0 :
                    cv2.putText(frame, roletext, (20, 55 + (cnt * 25)), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255))
                else :
                    cv2.putText(frame, roletext, (20, 55 + (cnt * 25)), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 0, 0))
                cnt += 1

        cv2.imshow('predicted output', frame)
        k == cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    else :
        print('error')