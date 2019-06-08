import os
import json

from typing import Dict, List, Tuple, Set, Optional

from lib.preprocessings import Chinese_selection_preprocessing
from lib.dataloaders import Selection_Dataset, Selection_loader
from lib.config import Hyper


class Runner(object):
    def __init__(self):
        self.hyper = Hyper('experiments/chinese_selection_re.json')
        self.preprocessor = Chinese_selection_preprocessing(self.hyper)

    def preprocessing(self):
        self.preprocessor.gen_relation_vocab()
        self.preprocessor.gen_all_data()
        self.preprocessor.gen_vocab(min_freq=1)
        # for ner only
        self.preprocessor.gen_bio_vocab()

    def run(self):
        print(self.hyper.__dict__)
        self.train()
        # self.preprocessing()

    def train(self):
        train_set = Selection_Dataset(self.hyper, self.hyper.train)
        loader = Selection_loader(train_set, batch_size=2, pin_memory=True)
        for batch_ndx, sample in enumerate(loader):
            # print(batch_ndx)
            # print(sample)
            print(sample.selection_id.size())
            print(sample.tokens_id.size())
            print(sample.bio_id.size())
            
            print(sample.spo_gold)
            print(sample.text)
            print(sample.bio)
            exit()


if __name__ == "__main__":
    config = Runner()
    config.run()