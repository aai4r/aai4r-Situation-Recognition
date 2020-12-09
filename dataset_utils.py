from utils import *
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import numpy as np
import json


max_num_nodes = 6

batch_size = 64
hidden_dimension = 1024
use_cuda = True
learning_rate = 1e-3
start_epoch = 0
end_epoch = 20
T = 4  # Number of propogation steps for the GGNN


imsitu = json.load(open("Dataset/imsitu/imsitu_space.json"))
train = json.load(open("Dataset/imsitu/train_v1.json"))
test = json.load(open("Dataset/imsitu/test_v1.json"))
v1_list = ['dining', 'eating', 'drinking', 'queuing', 'buying', 'paying',
           'serving', 'complaining', 'signaling', 'celebrating', 'cleaning', 'mopping']

nouns = imsitu["nouns"]
verbs = {}
for k, v in imsitu["verbs"].items():
    for v1 in v1_list:
        if k == v1:
            verbs[k] = v

# Adding a mapping for empty string since some of the nouns for roles are empty
nouns[''] = {'def': '', 'gloss': ['']}

# Load the verb vocabulary
with open('verb_vocabulary_v1.pickle', 'rb') as handle:
    verb_vocabulary, verb2index, verb2roles, verb2roles_with_ids = pickle.load(
        handle)
    verb_vocabulary_size = len(verb_vocabulary)
    verb_one_hot_embedding = np.eye(verb_vocabulary_size)

# Load the role vocabulary
with open('role_vocabulary_v1.pickle', 'rb') as handle:
    role_vocabulary, role2index = pickle.load(handle)
    role_vocabulary_size = len(role_vocabulary)
    role_one_hot_embedding = np.eye(role_vocabulary_size)

# Load the noun vocabulary
with open('noun_vocabulary_v1.pickle', 'rb') as handle:
    noun_vocabulary, noun2index = pickle.load(handle)
    noun_vocabulary_size = len(noun_vocabulary)

# Creating the verb to graph mapping (Returns graphs in format as required by build_graph function in GGNN)
verb2role_graph = build_graph(verb2roles, verb2index)
