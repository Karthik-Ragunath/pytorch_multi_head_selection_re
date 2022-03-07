import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np


def create_graph_data(node_embeddings=None, edge_matrix=None, batch_size=1):
    edge_matrix_axis_swapped = torch.swapaxes(edge_matrix, -1, -2)
    stacked_source_dest_tensors = torch.tensor([[], []])
    stacked_weight_tensors = torch.tensor([[]])
    for r_m in range(len(edge_matrix_axis_swapped)):
        source_dest_tensors = torch.tensor([[],[]])
        weight_tensors = torch.tensor([])
        for source in range(len(edge_matrix_axis_swapped[r_m])):
            for destination in range(len(edge_matrix_axis_swapped[r_m][source])):
                for relation in range(len(edge_matrix_axis_swapped[r_m][source][destination])):
                    source_dest_tensors = torch.cat((source_dest_tensors, torch.reshape(torch.tensor([source, destination]), (2, 1))), 1)
                    weight_tensors = torch.cat((weight_tensors, torch.tensor([edge_matrix_axis_swapped[r_m][source][destination][relation]])), 0)
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


def create_graph_encoding(node_embeddings=None, edge_matrix=None):
    batched_dataset = create_graph_data(node_embeddings=node_embeddings, edge_matrix=edge_matrix)
    graph_encodings = []
    for dataset in batched_dataset:
        gcn_conv = nn.GCNConv(in_channels=dataset.num_features, out_channels=300, add_self_loops=False)
        x = gcn_conv(x = dataset.x, edge_index=dataset.edge_index, edge_weight=dataset.edge_weights)
        graph_encodings.append(x)
    return graph_encodings