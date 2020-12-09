import time
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models

from utils import AverageMeter
from dataset_utils import *


def validate_model(test_loader, gnn, verb_model, verb_feature_model, noun_model, criterion):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    total_correct_verb = 0
    top5_total_correct_verb = 0
    total = 0

    total_correct = 0
    top5_total_correct = 0
    total_roles = 0

    value_all = 0
    top5_value_all = 0
    # total

    print_every = 100

    gnn.eval()
    end = time.time()
    with torch.no_grad():
        for i, (input_image, target, im_verb) in enumerate(test_loader):

            data_time.update(time.time() - end)

            target = target.cuda(async=True)

            input_var = torch.autograd.Variable(input_image).cuda()
            target_var = torch.autograd.Variable(target).long().cuda()

            verb_features = verb_feature_model(input_var)
            noun_features = noun_model(input_var)

            # transfom noun and verb_features to hidden representation
            hidden_verb_features = gnn.weight_iv(verb_features)
            hidden_noun_features = gnn.weight_in(noun_features).unsqueeze(1)

            im_verb = im_verb.squeeze(2).squeeze(1)
            # predict the verb of the given image
            if verb_model is not None:
                # image_verbs = torch.max(verb_model(input_var), 1)
                image_verbs = verb_model(input_var).argsort(
                    1, descending=True).cpu()
                total_correct_verb += image_verbs[
                    :, :1].eq(im_verb.long()).sum(1).sum(0).numpy()
                top5_total_correct_verb += image_verbs[
                    :, :5].eq(im_verb.long()).sum(1).sum(0).numpy()
                image_verbs = image_verbs[0, :1]
            else:
                image_verbs = im_verb
                total_correct_verb += image_verbs.eq(
                    im_verb).long().sum().numpy()
                top5_total_correct_verb += image_verbs.eq(
                    im_verb).long().sum().numpy()
            image_verbs = image_verbs.data.cpu().numpy()

            image_frame = [verb2roles_with_ids[x] for x in image_verbs]
            image_roles = [[role2index[x] for x in frame]
                           for frame in image_frame]
            image_roles = np.array([i + [0]*(max_num_nodes-len(i))
                                    for i in image_roles])

            role_graphs = []
            for verb in image_verbs:
                role_graphs.append(verb2role_graph[verb])

            role_adj_matrix, role_mask = gnn.build_adj_matrix(role_graphs)

            # creating the intial states with dimension BS x max-nodes x D
            init_state = np.zeros(
                (batch_size, max_num_nodes, hidden_dimension))

            W_v = gnn.verb_embedding(torch.from_numpy(
                image_verbs).cuda()).unsqueeze(1)
            W_e = gnn.role_embedding(torch.from_numpy(image_roles).cuda())

            init_states = gnn.inside_active(hidden_noun_features*W_e*W_v)

            # forward pass
            output = gnn(init_states, role_adj_matrix, role_mask, n_steps=T)
            output = gnn.classifier(output)

            loss = 0

            role_mask = role_mask.squeeze(2)
            for an_idx in range(0, 3):
                # forward pass
                # print(an_idx)
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

            correct, top5_correct, num_roles = accuracy_eval(
                output, target, role_mask)

            total_correct += correct
            top5_total_correct += top5_correct
            total_roles += num_roles
            total += len(output)

            acc = total_correct/total_roles
            top5_acc = top5_total_correct/total_roles

            if correct/num_roles == 1:
                value_all += 1
            if top5_correct/num_roles == 1:
                top5_value_all += 1

            batch_time.update(time.time() - end)
            end = time.time()

            losses.update(loss.data, input_var.size(0))

            if i % print_every == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                          i, len(test_loader),
                          batch_time=batch_time,
                          loss=losses,
                      ))
                print('Accuracy: {0:.4f} (Mean accuracy: {1:.4f}) (Top5 mean accuracy: {2:.4f}):'.format(
                    correct/num_roles, acc, top5_acc))
                print('Verb accuracy : {0:.4f} Top-5 verb accuracy : {1:.4f}'.format(
                    total_correct_verb / total, top5_total_correct_verb / total))
                print('Value all accuracy : {0:.4f} Top-5 value all accuracy : {1:.4f}'.format(
                    value_all / total, top5_value_all / total))
                # print("total correct : {0} , total roles : {1}".format(
                #     total_correct, total_roles))
    return total_correct_verb / total, acc, value_all / total, top5_total_correct_verb / total, top5_acc, top5_value_all/total


def accuracy_eval(prediction, target, mask):
    # prediction is of size  max_num_nodes x vocabulary size
    # target is of size  max_num_nodes

    prediction = prediction.squeeze(0)
    target = target.squeeze(0)
    # _, predicted_label = torch.max(prediction, 1)
    predicted_label = prediction.argsort(1, descending=True)
    # print(predicted_label.shape)
    # print(target.shape)
    #
    # print(target)
    # print(predicted_label)

    target_an = target[0, :]
    correct = (target_an.reshape(-1, 1).eq(
        predicted_label[:, :1]).long().sum(1)*mask.long()).cpu().numpy()
    top5_correct = (target_an.reshape(-1, 1).eq(
        predicted_label[:, :5]).long().sum(1)*mask.long()).cpu().numpy()
    total = torch.sum(mask).cpu().numpy()
    # print(correct)
    # print(total)

    for an_idx in range(1, 3):
        target_an = target[an_idx, :]
        correct_temp = (target_an.reshape(-1, 1).eq(
            predicted_label[:, :1]).long().sum(1)*mask.long()).cpu().numpy()
        top5_correct_temp = (target_an.reshape(-1, 1).eq(
            predicted_label[:, :5]).long().sum(1)*mask.long()).cpu().numpy()
        correct += correct_temp
        top5_correct += top5_correct_temp

    _, correct = correct.nonzero()
    _, top5_correct = top5_correct.nonzero()

    return correct.shape[0], top5_correct.shape[0], total
