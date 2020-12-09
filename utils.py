import numpy as np
import torch
import shutil

from torch.autograd import Variable


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def index2noun(noun_dict, index):
    '''Function to get the noun of a given noun id'''
    return noun_dict[index]['gloss'][0]


def load_image_verb_and_role(image_dict,
                             image_key,
                             verb_embedding,
                             role_embedding,
                             verb2index,
                             role2index):
    '''Function to get the verb and role encoding'''
    image_verb = image_dict[image_key]['verb']
    image_frame = image_dict[image_key]['frames'][0]
    image_roles = list(image_frame.keys())

    verb_one_hot = verb_embedding[verb2index[image_verb]]
    role_one_hot = [role_embedding[role2index[image_role]]
                    for image_role in image_roles]

    return verb_one_hot, role_one_hot


def to_var(x, volatile=False, requires_grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile, requires_grad=requires_grad)

# Function to build a fully connected graph for each of the verbs and create a mapping between them


def build_graph(verb2roles, verb2index):

    verb2role_graph = dict()

    for key, values in verb2roles.items():

        edge_type = 0  # for undirected edge

        graph_temp = {}
        graph_temp['num_nodes'] = len(values)
        nodes = list(range(0, len(values)))
        graph_temp['edges'] = {edge_type: [
            (x, y) for x in nodes for y in nodes if x != y]}

        verb2role_graph[verb2index[key]] = graph_temp

    return verb2role_graph
