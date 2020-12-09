import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from utils import save_checkpoint, index2noun, to_var


max_num_nodes = 6


class GNN_batch(nn.Module):
    def __init__(self, noun_vocabulary_size, verb_vocabulary_size, role_vocabulary_size, hidden_dim, feature_length=4096, n_edges=1):
        '''
        :Note: this implementation is only for undirected graphs.
        Does NOT support multiple edges.
        :param configs:
        :hidden_dim: the dimension of the hidden state
        :edge_types: the number of edge types
        '''
        super(GNN_batch, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_edges = n_edges
        self.feature_length = feature_length
        self.output_dim = noun_vocabulary_size

        # different edge has different weights
        self._prop_net = nn.Linear(hidden_dim, hidden_dim * n_edges)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.transform = nn.Linear(hidden_dim * 2, hidden_dim)
        # self.transform = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU())

        # For graph-level prediction
        self.i_func = nn.Sequential(nn.Linear(
            hidden_dim*2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.j_func = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attention = nn.Sigmoid()
        self.inside_active = nn.Tanh()
        self.outside_active = nn.Tanh()

        # embedding matrices
        self.verb_embedding = nn.Embedding(verb_vocabulary_size, hidden_dim)
        self.role_embedding = nn.Embedding(role_vocabulary_size, hidden_dim)

        # feture vector transformation
        self.weight_iv = nn.Linear(self.feature_length, self.hidden_dim)
        self.weight_in = nn.Linear(self.feature_length, self.hidden_dim)

        # Layer to classify noun from hidden state
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim)

    @staticmethod
    def build_adj_matrix(graphs, n_edges=1):
        '''
        :param graphs: a list of graphs (each graph is a dictionary)
        'e.g. { 'num_nodes': 3,
                'edges':{0: [(0,1), (1, 0)]} } edge_type: edges
        '
        :return: a list of adjacent matrices and hidden mask
        '''

        num_graphs = len(graphs)

        # adjacency matrices with max size as max number of nodes across all graphs
        batch_adj_matrices = to_var(torch.zeros(
            num_graphs, max_num_nodes, max_num_nodes * n_edges))
        # mask - BS x MaxNodes (to kill hidden states for not existing nodes)
        # third dim present to apply mask easily :)
        batch_mask = np.ones((num_graphs, max_num_nodes, 1), dtype='float32')
        for g, graph in enumerate(graphs):
            num_nodes = graph['num_nodes']
            edges = graph['edges']
            for e_type, links in edges.items():
                for src, dst in links:
                    batch_adj_matrices[g, dst, e_type*max_num_nodes + src] = 1

            # set things in mask to 0
            batch_mask[g, num_nodes:] = 0
        return batch_adj_matrices, to_var(torch.from_numpy(batch_mask))

    def prop_net(self, hidden_states):
        return self._prop_net(hidden_states)

    def gating(self, messages, hiddens):
        D = self.hidden_dim
        joined_input = torch.cat([messages, hiddens], 2)
        gated_results = nn.Sigmoid()(self.gate(joined_input))
        z_gate = gated_results.narrow(2, 0, D)
        r_gate = gated_results.narrow(2, D, D)
        h_tilde = nn.Tanh()(self.transform(
            torch.cat([messages, r_gate * hiddens], dim=2)))
        output = hiddens + z_gate * (h_tilde - hiddens)
        return output

    def forward(self, init_states, adj_matrix, mask, n_steps):
        '''
        :param init_states: the initial hidden states of each node; a 3D tensor BS x max-nodes x D
        :param adj_matrix: the adjacent matrices; a 3D tensor BS x max-nodes x max-nodes
        :param mask: masking for hiddens; a 3D matrix BS x max-nodes x 1
        :param n_steps: the number of propagation steps
        :return: the hidden states of each node; should be consistent with init_states
        '''
        D = self.hidden_dim
        inputs = init_states.clone().cuda()
        for i_step in range(n_steps):
            # BS x max-num-nodes x (D * n_edges)
            propagate_feats = self.prop_net(inputs)

            if self.n_edges > 1:
                propagate_feats = torch.cat(
                    [propagate_feats[:, :, idx*D:(idx+1)*D] for idx in range(self.n_edges)], 1)
            # BS x max-num-nodes * n_edges x D, each edge is grouped
            # BS x max-num-nodes x D
            messages = torch.bmm(adj_matrix, propagate_feats)
            messages = messages * mask  # mask messages!
            messages = messages / \
                (torch.sum(adj_matrix, dim=2, keepdim=True) + 1e-6)  # normalization
            outputs = self.gating(messages, inputs)

            inputs = outputs * mask  # mask hiddens!

        return inputs
        # inputs: node-level hidden-states
