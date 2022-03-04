import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import json
import os
import copy

from typing import Dict, List, Tuple, Set, Optional
from functools import partial

from torchcrf import CRF
from pytorch_transformers import *

import sys
import numpy as np
from contextlib import redirect_stdout


class MultiHeadSelection(nn.Module):
    def __init__(self, hyper) -> None:
        super(MultiHeadSelection, self).__init__()

        # block to write outputs to a file
        original_stdout = sys.stdout
        if not os.path.exists('multi_head_model_info.txt'):
            open('multi_head_model_info.txt', 'w+').close()

        self.print_once_forward = True

        with open('multi_head_model_info.txt', 'w') as fp:
            sys.stdout = fp
            self.hyper = hyper
            self.data_root = hyper.data_root
            self.gpu = hyper.gpu

            self.word_vocab = json.load(
                open(os.path.join(self.data_root, 'word_vocab.json'), 'r'))
            self.relation_vocab = json.load(
                open(os.path.join(self.data_root, 'relation_vocab.json'), 'r'))
            self.bio_vocab = json.load(
                open(os.path.join(self.data_root, 'bio_vocab.json'), 'r'))
            self.id2bio = {v: k for k, v in self.bio_vocab.items()}

            self.word_embeddings = nn.Embedding(num_embeddings=len(
                self.word_vocab),
                embedding_dim=hyper.emb_size)
            print("Word Embedding:", "num_embeddings =", len(self.word_vocab), "embedding_dim =", hyper.emb_size)

            self.relation_emb = nn.Embedding(num_embeddings=len(
                self.relation_vocab),
                embedding_dim=hyper.rel_emb_size)
            print("Relation Embedding:", "num_embeddings =", len(self.relation_vocab), "embedding_dim =", hyper.rel_emb_size)
            # bio + pad
            self.bio_emb = nn.Embedding(num_embeddings=len(self.bio_vocab),
                                        embedding_dim=hyper.bio_emb_size)
            print("Bio Embedding:", "num_embedding =", len(self.bio_vocab), "embedding_dim =", hyper.bio_emb_size)

            print("LSTM or GRU:", "input_size =", hyper.emb_size, "hidden_size =", hyper.hidden_size)
            print("---------Batch First is enabled----------")
            print("Input Size:", "(Batch_Size, Sequence_Length, Input_Size)")
            print("Output Dimension:", "(Batch_Size, Sequence_Length, Num_Dimensions * Output_Size)")
            if hyper.cell_name == 'gru':
                self.encoder = nn.GRU(hyper.emb_size,
                                      hyper.hidden_size,
                                      bidirectional=True,
                                      batch_first=True)
            elif hyper.cell_name == 'lstm':
                # hyper.emb_size -> input size (usually one hot encoding)
                # hyper.hidden_size -> output vector dimension from each LSTM cell
                # bidirectional -> respresents directions in play
                # actual input dimension -> (batch_size, sequence_len, hyper.emb_size)
                # actual hidden output dimension -> (batch_size, num_dirns * num_layers, hyper.hidden_size)
                # actual cell output -> (batch_size, num_dirns * num_layers, hyper.proj_size or hyoer.hidden_size)
                self.encoder = nn.LSTM(hyper.emb_size,
                                       hyper.hidden_size,
                                       bidirectional=True,
                                       batch_first=True)
            elif hyper.cell_name == 'bert':
                self.post_lstm = nn.LSTM(hyper.emb_size,
                                       hyper.hidden_size,
                                       bidirectional=True,
                                       batch_first=True)
                self.encoder = BertModel.from_pretrained('bert-base-uncased')
                for name, param in self.encoder.named_parameters():
                    # 11 actually indicates final head block of the multi headed bert model
                    if '11' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                        # print(name, param.size())
            else:
                raise ValueError('cell name should be gru/lstm/bert!')
            
            if hyper.activation.lower() == 'relu':
                self.activation = nn.ReLU()
            elif hyper.activation.lower() == 'tanh':
                self.activation = nn.Tanh()
            else:
                raise ValueError('unexpected activation!')

            self.tagger = CRF(len(self.bio_vocab) - 1, batch_first=True) # CRF(3, batchFirst=True)

            self.selection_u = nn.Linear(hyper.hidden_size + hyper.bio_emb_size,
                                         hyper.rel_emb_size) # (300 + 50), 100
            self.selection_v = nn.Linear(hyper.hidden_size + hyper.bio_emb_size,
                                         hyper.rel_emb_size)
            self.selection_uv = nn.Linear(2 * hyper.rel_emb_size,
                                          hyper.rel_emb_size)
            self.emission = nn.Linear(hyper.hidden_size, len(self.bio_vocab) - 1) # nn.linear(300, 3)

            self.bert2hidden = nn.Linear(768, hyper.hidden_size)
            # for bert_lstm
            # self.bert2hidden = nn.Linear(768, hyper.emb_size)
            
            print("Selection U Linear:", hyper.hidden_size + hyper.bio_emb_size, hyper.rel_emb_size)
            print("Selection V Linear:", hyper.hidden_size + hyper.bio_emb_size, hyper.rel_emb_size)
            print("Selection UV Linear:", 2 * hyper.rel_emb_size, hyper.rel_emb_size)
            print("Emission:", hyper.hidden_size, len(self.bio_vocab) - 1)
            print("Bert2Hidden:", 768, hyper.hidden_size)

            if self.hyper.cell_name == 'bert':

                self.bert_tokenizer = BertTokenizer.from_pretrained(
                    'bert-base-uncased')

            # self.accuracy = F1Selection()
            sys.stdout = original_stdout

    def inference(self, mask, text_list, decoded_tag, selection_logits):
        selection_mask = (mask.unsqueeze(2) *
                          mask.unsqueeze(1)).unsqueeze(2).expand(
                              -1, -1, len(self.relation_vocab),
                              -1)  # batch x seq x rel x seq
        selection_tags = (torch.sigmoid(selection_logits) *
                          selection_mask.float()) > self.hyper.threshold

        selection_triplets = self.selection_decode(text_list, decoded_tag,
                                                   selection_tags)

        if self.print_once_forward:
            with open('forward_info.txt', 'a') as fp:
                print("Selection Mask:", np.shape(selection_mask), file=fp)
                print("Selection Tags:", np.shape(selection_tags), file=fp)
                print("Selection Triplets:", np.shape(selection_triplets), file=fp)
        return selection_triplets

    def masked_BCEloss(self, mask, selection_logits, selection_gold):
        selection_mask = (mask.unsqueeze(2) *
                          mask.unsqueeze(1)).unsqueeze(2).expand(
                              -1, -1, len(self.relation_vocab),
                              -1)  # batch x seq x rel x seq
        selection_loss = F.binary_cross_entropy_with_logits(selection_logits,
                                                            selection_gold,
                                                            reduction='none') 
        if self.print_once_forward:
            with open('forward_info.txt', 'a') as fp:
                print("Selection Mask:", np.shape(selection_mask), file=fp)
                print("Selection Gold:", np.shape(selection_gold), file=fp)
                print("Selection Loss Pre Sum:", np.shape(selection_loss), file=fp)
        selection_loss = selection_loss.masked_select(selection_mask).sum()
        selection_loss /= mask.sum()
        return selection_loss


    @staticmethod
    def description(epoch, epoch_num, output):
        return "L: {:.2f}, L_crf: {:.2f}, L_selection: {:.2f}, epoch: {}/{}:".format(
            output['loss'].item(), output['crf_loss'].item(),
            output['selection_loss'].item(), epoch, epoch_num)


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


    def forward(self, sample, is_train: bool) -> Dict[str, torch.Tensor]:
        # Forward method called
        tokens = sample.tokens_id.cuda(self.gpu) # text tokens
        selection_gold = sample.selection_id.cuda(self.gpu) # relation - N * seq_len * rel_size * seq_len
        bio_gold = sample.bio_id.cuda(self.gpu) # N * seq_len

        text_list = sample.text # N * seq_len
        spo_gold = sample.spo_gold # N 

        bio_text = sample.bio
        if self.print_once_forward:
            print("*" * 50, "Inside", "*" * 50)
            if not os.path.exists('forward_info.txt'):
                open('foward_info.txt', 'w+').close()
            with open('forward_info.txt', 'w') as fp:
                #with redirect_stdout(fp):
                print("Tokens:", tokens, file=fp)
                print("Tokens Shape:", np.shape(tokens), file=fp)
                print("Selection Gold:", selection_gold, file=fp)
                print("Selection Gold Shape:", np.shape(selection_gold), file=fp)
                print("bio_gold:", bio_gold, file=fp)
                print("BIO Gold Shape:", np.shape(bio_gold), file=fp)
                print("text_list:", text_list, file=fp)
                print("Text List Shape:", np.shape(text_list), file=fp)
                print("spo_gold:", spo_gold, file=fp)
                print("SPO Gold Shape:", np.shape(spo_gold), file=fp)
                print("bio_text:", bio_text, file=fp)
                print("BIO Text Shape:", np.shape(bio_text), file=fp)

        if self.hyper.cell_name in ('gru', 'lstm'):
            mask = tokens != self.word_vocab['<pad>']  # batch x seq
            bio_mask = mask
        elif self.hyper.cell_name in ('bert'):
            notpad = tokens != self.bert_tokenizer.encode('[PAD]')[0]
            notcls = tokens != self.bert_tokenizer.encode('[CLS]')[0]
            notsep = tokens != self.bert_tokenizer.encode('[SEQ]')[0]
            mask = notpad & notcls & notsep
            bio_mask = notpad & notsep  # fst token for crf cannot be masked
        else:
            raise ValueError('unexpected encoder name!')

        if self.print_once_forward:
            with open('forward_info.txt', 'a') as fp:
                #with redirect_stdout(fp):
                print("Mask:", mask, file=fp)
                print("Mask Shape:", np.shape(mask), file=fp)
                print("Notpad:", notpad, file=fp)
                print("Notpad Shape:", np.shape(notpad), file=fp)
                print("Notcls:", notcls, file=fp)
                print("Notcls Shape:", np.shape(notcls), file=fp)
                print("Notsep:", notsep, file=fp)
                print("Notsep Shape:", np.shape(notsep), file=fp)
                print("BioMask:", bio_mask, file=fp)
                print("BioMask Shape:", np.shape(bio_mask), file=fp)

        if self.hyper.cell_name in ('lstm', 'gru'):
            embedded = self.word_embeddings(tokens)
            o, h = self.encoder(embedded)

            o = (lambda a: sum(a) / 2)(torch.split(o,
                                                   self.hyper.hidden_size,
                                                   dim=2))
        elif self.hyper.cell_name == 'bert':
            # with torch.no_grad():
            o = self.encoder(tokens, attention_mask=mask)[
                0]  # last hidden of BERT

            if self.print_once_forward:
                with open('forward_info.txt', 'a') as fp:
                    #with redirect_stdout(fp):
                    print("Bert Encoder Output Shape:", np.shape(o), file=fp)

            # o = self.activation(o)
            # torch.Size([16, 310, 768])
            o = self.bert2hidden(o)

            if self.print_once_forward:
                with open('forward_info.txt', 'a') as fp:
                    #with redirect_stdout(fp):
                    print("Bert2Hidden Output Shape:", np.shape(o), file=fp)

            # below for bert+lstm
            # o, h = self.post_lstm(o)

            # o = (lambda a: sum(a) / 2)(torch.split(o,
            #                                        self.hyper.hidden_size,
            #                                        dim=2))
        else:
            raise ValueError('unexpected encoder name!')
        emi = self.emission(o)

        output = {}

        crf_loss = 0

        if is_train:
            crf_loss = -self.tagger(emi, bio_gold,
                                    mask=bio_mask, reduction='mean')
            # (14, 250, 3), (14, 250)
        else:
            decoded_tag = self.tagger.decode(emissions=emi, mask=bio_mask)

            output['decoded_tag'] = [list(map(lambda x : self.id2bio[x], tags)) for tags in decoded_tag]
            output['gold_tags'] = bio_text

            temp_tag = copy.deepcopy(decoded_tag)
            for line in temp_tag:
                line.extend([self.bio_vocab['<pad>']] *
                            (self.hyper.max_text_len - len(line)))
            bio_gold = torch.tensor(temp_tag).cuda(self.gpu)

        tag_emb = self.bio_emb(bio_gold)

        o = torch.cat((o, tag_emb), dim=2)

        if self.print_once_forward:
            with open('forward_info.txt', 'a') as fp:
                #with redirect_stdout(fp):
                print("Emission Shape:", np.shape(emi), file=fp)
                print("CRF Loss:", crf_loss, file=fp)
                print("Tag Embedding:", np.shape(tag_emb), file=fp)
                print("Torch Cat (O):", np.shape(o), file=fp)

        # forward multi head selection
        B, L, H = o.size()
        u = self.activation(self.selection_u(o)).unsqueeze(1).expand(B, L, L, -1)
        v = self.activation(self.selection_v(o)).unsqueeze(2).expand(B, L, L, -1)
        uv = self.activation(self.selection_uv(torch.cat((u, v), dim=-1)))

        # correct one
        selection_logits = torch.einsum('bijh,rh->birj', uv,
                                        self.relation_emb.weight)

        # Pass it to Neural Network

        if self.print_once_forward:
            with open('forward_info.txt', 'a') as fp:
                #with redirect_stdout(fp):
                print("B,L,H:", B, L, H, file=fp)
                print("U Shape:", np.shape(u), file=fp)
                print("V Shape:", np.shape(v), file=fp)
                print("UV Shape:", np.shape(uv), file=fp)
                print("Selection Logits:", np.shape(selection_logits), file=fp)

        graph_encodings = create_graph_encoding(node_embeddings=o, edge_matrix=selection_logits)
        # must use this graph_encoding as lstm input or perform one more level of graph convolutions
        # and pass their outputs as inputs to decoder described in "Get To The Point Summarizer" 

        # use loop instead of matrix
        # selection_logits_list = []
        # for i in range(self.hyper.max_text_len):
        #     uvi = uv[:, i, :, :]
        #     sigmoid_input = uvi
        #     selection_logits_i = torch.einsum('bjh,rh->brj', sigmoid_input,
        #                                         self.relation_emb.weight).unsqueeze(1)
        #     selection_logits_list.append(selection_logits_i)
        # selection_logits = torch.cat(selection_logits_list,dim=1)


        if not is_train:
            output['selection_triplets'] = self.inference(
                mask, text_list, decoded_tag, selection_logits)
            output['spo_gold'] = spo_gold

        selection_loss = 0
        if is_train:
            selection_loss = self.masked_BCEloss(mask, selection_logits,
                                                 selection_gold)

        loss = crf_loss + selection_loss
        output['crf_loss'] = crf_loss
        output['selection_loss'] = selection_loss
        output['loss'] = loss

        output['description'] = partial(self.description, output=output)

        if self.print_once_forward:
            with open('forward_info.txt', 'a') as fp:
                #with redirect_stdout(fp):
                print("Selection Loss:", selection_loss, file=fp)
                print("Loss:", loss, file=fp)
                print("Output:", output, file=fp)
        return output

    def selection_decode(self, text_list, sequence_tags,
                         selection_tags: torch.Tensor
                         ) -> List[List[Dict[str, str]]]:
        reversed_relation_vocab = {
            v: k
            for k, v in self.relation_vocab.items()
        }

        reversed_bio_vocab = {v: k for k, v in self.bio_vocab.items()}

        text_list = list(map(list, text_list))

        def find_entity(pos, text, sequence_tags):
            entity = []

            if sequence_tags[pos] in ('B', 'O'):
                entity.append(text[pos])
            else:
                temp_entity = []
                while sequence_tags[pos] == 'I':
                    temp_entity.append(text[pos])
                    pos -= 1
                    if pos < 0:
                        break
                    if sequence_tags[pos] == 'B':
                        temp_entity.append(text[pos])
                        break
                entity = list(reversed(temp_entity))
            return ''.join(entity)

        batch_num = len(sequence_tags)
        result = [[] for _ in range(batch_num)]
        idx = torch.nonzero(selection_tags.cpu())
        for i in range(idx.size(0)):
            b, s, p, o = idx[i].tolist()

            predicate = reversed_relation_vocab[p]
            if predicate == 'N':
                continue
            tags = list(map(lambda x: reversed_bio_vocab[x], sequence_tags[b]))
            object = find_entity(o, text_list[b], tags)
            subject = find_entity(s, text_list[b], tags)

            assert object != '' and subject != ''

            triplet = {
                'object': object,
                'predicate': predicate,
                'subject': subject
            }
            result[b].append(triplet)
        return result

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        pass
