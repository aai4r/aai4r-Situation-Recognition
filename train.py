import time
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models

from utils import AverageMeter
from dataset_utils import *


def train_model(train_loader, gnn, verb_feature_model, noun_model, criterion, optimizer, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    print_every = 20

    gnn.train()
    end = time.time()

    for i, (input_image, target, im_verb) in enumerate(train_loader):

        data_time.update(time.time() - end)
        # print(target.shape)
        target = target.cuda(async=True)

        # plt.imshow(temp.numpy()[0,:,:,:])
        # plt.show()

        input_var = torch.autograd.Variable(input_image).cuda()
        target_var = torch.autograd.Variable(target).long().cuda()

        verb_features = verb_feature_model(input_var)
        noun_features = noun_model(input_var)

        # transfom noun and verb_features to hidden representation
        hidden_verb_features = gnn.weight_iv(verb_features)
        hidden_noun_features = gnn.weight_in(noun_features).unsqueeze(1)

        im_verb = im_verb.squeeze(2).squeeze(1)
        image_verbs = im_verb.data.cpu().numpy()

        image_frame = [verb2roles_with_ids[x] for x in image_verbs]
        image_roles = [[role2index[x] for x in frame]for frame in image_frame]
        image_roles = np.array([i + [0]*(max_num_nodes-len(i))
                                for i in image_roles])

        role_graphs = []
        for verb in image_verbs:
            role_graphs.append(verb2role_graph[verb])

        role_adj_matrix, role_mask = gnn.build_adj_matrix(role_graphs)

        # creating the intial states with dimension BS x max-nodes x D
        init_state = np.zeros((batch_size, max_num_nodes, hidden_dimension))

        W_v = gnn.verb_embedding(torch.from_numpy(
            image_verbs).cuda()).unsqueeze(1)
        W_e = gnn.role_embedding(torch.from_numpy(image_roles).cuda())

        init_states = gnn.inside_active(hidden_noun_features*W_e*W_v)

        loss = 0

        role_mask = role_mask.squeeze(2)
        for an_idx in range(0, 3):

            output = gnn(init_states, role_adj_matrix,
                         role_mask.unsqueeze(2), n_steps=T)
            output = gnn.classifier(output)

            an_loss = 0
            for j in range(max_num_nodes):
                temp_loss = criterion(
                    output[:, j, :], target[:, an_idx, j])*role_mask[:, j]
                an_loss += temp_loss

            an_loss = torch.mean(an_loss)
            loss += an_loss

        # compute gradient and do rmsprop step
        optimizer.zero_grad()
        loss.backward()

        # clip the graident
        torch.nn.utils.clip_grad_norm(gnn.parameters(), 1)
        #gnn.parameters().grad.data.clamp_(-1, 1)
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.data, input_var.size(0))

        if i % print_every == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      epoch, i, len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses))
