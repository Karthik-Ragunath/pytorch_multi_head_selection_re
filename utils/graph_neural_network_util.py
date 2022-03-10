import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np


ddef create_graph_data(self, node_embeddings=None, edge_matrix=None, batch_size=1):
    ### Have to parallelize this block.
    edge_matrix_axis_swapped = torch.swapaxes(edge_matrix, -1, -2).cuda(self.gpu) 
    stacked_source_dest_tensors = torch.tensor([[], []]).cuda(self.gpu) 
    stacked_weight_tensors = torch.tensor([[]]).cuda(self.gpu)
    for r_m in range(len(edge_matrix_axis_swapped)):
        source_dest_tensors = torch.tensor([[],[]]).cuda(self.gpu)
        weight_tensors = torch.tensor([]).cuda(self.gpu)
        for source in range(len(edge_matrix_axis_swapped[r_m])):
            for destination in range(len(edge_matrix_axis_swapped[r_m][source])):
                for relation in range(len(edge_matrix_axis_swapped[r_m][source][destination])):
                    source_dest_tensors = torch.cat((source_dest_tensors, torch.reshape(torch.tensor([source, destination]).cuda(self.gpu), (2, 1))), 1)
                    weight_tensors = torch.cat((weight_tensors, torch.tensor([edge_matrix_axis_swapped[r_m][source][destination][relation]]).cuda(self.gpu)), 0)
        source_dest_tensors_unsqueezed = torch.unsqueeze(source_dest_tensors, dim=0)
        weight_tensors_unsqueezed = torch.unsqueeze(weight_tensors, dim=0)
        if not stacked_source_dest_tensors.numel():
            stacked_source_dest_tensors = source_dest_tensors_unsqueezed
            stacked_weight_tensors = weight_tensors_unsqueezed
        else:
            stacked_source_dest_tensors = torch.cat((stacked_source_dest_tensors, source_dest_tensors_unsqueezed), dim=0)
            stacked_weight_tensors = torch.cat((stacked_weight_tensors, weight_tensors_unsqueezed), dim=0)

    stacked_source_dest_tensors = stacked_source_dest_tensors.to(dtype=torch.int64)
    stacked_weight_tensors = stacked_weight_tensors.to(dtype=torch.float32)
    node_embeddings = node_embeddings.to(dtype=torch.float32)

    dataset_list = []
    for index in range(len(stacked_source_dest_tensors)):
        dataset = Data(x=node_embeddings[index], edge_index=stacked_source_dest_tensors[index], edge_weights=stacked_weight_tensors[index])
        dataset_list.append(dataset)
    batched_dataset = DataLoader(dataset_list, batch_size=batch_size)
    return batched_dataset


def create_graph_encoding(self, node_embeddings=None, edge_matrix=None, out_channel_size=300, batch_size=1):
    batched_dataset = self.create_graph_data(node_embeddings=node_embeddings, edge_matrix=edge_matrix, batch_size=batch_size)
    gcn_conv = torch_geometric.nn.GCNConv(in_channels=len(node_embeddings[0][0]), out_channels=out_channel_size, add_self_loops=False).cuda(self.gpu)
    graph_encodings = torch.tensor([]).cuda(self.gpu)
    for dataset in batched_dataset:
        x = gcn_conv(x = dataset.x, edge_index=dataset.edge_index, edge_weight=dataset.edge_weights)
        graph_encoding_reshaped = torch.reshape(x, (-1, len(node_embeddings[0]), out_channel_size))
        if not graph_encodings.numel():
            graph_encodings = graph_encoding_reshaped
        else:
            graph_encodings = torch.cat((graph_encodings, graph_encoding_reshaped), dim=0)
    return graph_encodings


def encode_graph_encodings(self, graph_encodings=None):
    batch_size = graph_encodings.shape[0]
    seq_len = graph_encodings.shape[1]
    features_dim = graph_encodings.shape[2]

    num_layers = 1
    num_directions = 2
    output_size = graph_encodings.shape[2]

    hidden = torch.randn(num_layers * num_directions, batch_size, output_size).cuda(self.gpu)
    cell_state = torch.randn(num_layers * num_directions, batch_size, output_size).cuda(self.gpu)
    cell = torch.nn.LSTM(input_size = features_dim, hidden_size = output_size, batch_first = True, num_layers = 1, bidirectional = True).cuda(self.gpu)

    out, hidden = cell(graph_encodings, (hidden, cell_state))
    return out[:, -1, :]