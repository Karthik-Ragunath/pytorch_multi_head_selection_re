import os
import json

import torch

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

from torch.nn.utils.rnn import pad_sequence

from functools import partial
from typing import Dict, List, Tuple, Set, Optional

from pytorch_transformers import *

import numpy as np

import sys

class Selection_Dataset(Dataset):
    def __init__(self, hyper, dataset):
        self.hyper = hyper
        self.data_root = hyper.data_root

        self.word_vocab = json.load(
            open(os.path.join(self.data_root, 'word_vocab.json'), 'r'))
        self.relation_vocab = json.load(
            open(os.path.join(self.data_root, 'relation_vocab.json'), 'r'))
        self.bio_vocab = json.load(
            open(os.path.join(self.data_root, 'bio_vocab.json'), 'r'))

        self.selection_list = []
        self.text_list = []
        self.bio_list = []
        self.spo_list = []

        # for bert only
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased')

        for line in open(os.path.join(self.data_root, dataset), 'r'):
            line = line.strip("\n")
            instance = json.loads(line)

            self.selection_list.append(instance['selection'])
            self.text_list.append(instance['text'])
            self.bio_list.append(instance['bio'])
            self.spo_list.append(instance['spo_list'])

    def __getitem__(self, index):
        selection = self.selection_list[index]
        text = self.text_list[index]
        bio = self.bio_list[index]
        spo = self.spo_list[index]
        if self.hyper.cell_name == 'bert':
            text, bio, selection = self.pad_bert(text, bio, selection)
            tokens_id = torch.tensor(
                self.bert_tokenizer.convert_tokens_to_ids(text))
        else:
            tokens_id = self.text2tensor(text)
        bio_id = self.bio2tensor(bio)
        selection_id = self.selection2tensor(text, selection)

        return tokens_id, bio_id, selection_id, len(text), spo, text, bio

    def __len__(self):
        return len(self.text_list)

    def pad_bert(self, text: List[str], bio: List[str], selection: List[Dict[str, int]]) -> Tuple[List[str], List[str], Dict[str, int]]:
        # for [CLS] and [SEP]
        text = ['[CLS]'] + text + ['[SEP]']
        bio = ['O'] + bio + ['O']
        selection = [{'subject': triplet['subject'] + 1, 'object': triplet['object'] +
                      1, 'predicate': triplet['predicate']} for triplet in selection]
        assert len(text) <= self.hyper.max_text_len
        text = text + ['[PAD]'] * (self.hyper.max_text_len - len(text))
        return text, bio, selection

    def text2tensor(self, text: List[str]) -> torch.tensor:
        # TODO: tokenizer
        oov = self.word_vocab['oov']
        padded_list = list(map(lambda x: self.word_vocab.get(x, oov), text))
        padded_list.extend([self.word_vocab['<pad>']] *
                           (self.hyper.max_text_len - len(text)))
        return torch.tensor(padded_list)

    def bio2tensor(self, bio):
        # here we pad bio with "O". Then, in our model, we will mask this "O" padding.
        # in multi-head selection, we will use "<pad>" token embedding instead.
        padded_list = list(map(lambda x: self.bio_vocab[x], bio))
        padded_list.extend([self.bio_vocab['O']] *
                           (self.hyper.max_text_len - len(bio)))
        return torch.tensor(padded_list)

    def selection2tensor(self, text, selection):
        # s p o
        result = torch.zeros(
            (self.hyper.max_text_len, len(self.relation_vocab),
             self.hyper.max_text_len))
        NA = self.relation_vocab['N']
        result[:, NA, :] = 1
        for triplet in selection:

            object = triplet['object']
            subject = triplet['subject']
            predicate = triplet['predicate']

            result[subject, predicate, object] = 1
            result[subject, NA, object] = 0

        return result


class Batch_reader(object):
    def __init__(self, data):
        transposed_data = list(zip(*data))
        # tokens_id, bio_id, selection_id, spo, text, bio
        original_stdout = sys.stdout
        if os.path.exists('batch_data_info.txt'):
            open('batch_data_info.txt', 'w+').close()
        
        with open('batch_data_info.txt', 'w') as fp:
            sys.stdout = fp
            print("================= Batch Functionality ==================")
            print("tokens_id before padding:", np.ndim(transposed_data[0]), np.shape(transposed_data[0]))
            self.tokens_id = pad_sequence(transposed_data[0], batch_first=True)
            print("tokens_id after padding:", np.ndim(self.tokens_id), np.shape(self.tokens_id))
            print("bio_id before padding:", np.ndim(transposed_data[1]), np.shape(transposed_data[1])) # Tuple of tensors, each tensor is of dim - 250 => (14,)
            self.bio_id = pad_sequence(transposed_data[1], batch_first=True)
            print("bio_id after padding:", np.ndim(self.bio_id), np.shape(self.bio_id)) # tensor of list of list => (14 * 250)
            print("selection_id before padding:", np.ndim(transposed_data[2]), np.shape(transposed_data[2]))
            self.selection_id = torch.stack(transposed_data[2], 0)
            print("selection_id after padding:", np.ndim(self.selection_id), np.shape(self.selection_id))
            self.length = transposed_data[3]
            print("Length:", transposed_data[3])
            self.spo_gold = transposed_data[4]
            print("SPO_GOLD (tranposed_data[4]):", np.ndim(transposed_data[4]), np.shape(transposed_data[4]))
            self.text = transposed_data[5]
            print("text (tranposed_data[5]):", np.ndim(transposed_data[5]), np.shape(transposed_data[5]))
            self.bio = transposed_data[6]
            print("bio (tranposed_data[6]):", np.ndim(transposed_data[6]), np.shape(transposed_data[6]))
            print("=========================================================")
            sys.stdout = original_stdout

    def pin_memory(self):
        # custom data must be pinned in this way in order to instruct the compiler to be page locked
        self.tokens_id = self.tokens_id.pin_memory()
        self.bio_id = self.bio_id.pin_memory()
        self.selection_id = self.selection_id.pin_memory()
        return self


def collate_fn(batch):
    return Batch_reader(batch)


Selection_loader = partial(DataLoader, collate_fn=collate_fn, pin_memory=True) # Dataset is the positional argument passes from the place it is called
