# "Joint entity recognition and relation extraction as a multi-head selection problem" (Expert Syst. Appl, 2018)

This repository is forked from [link](https://github.com/WindChimeRan/pytorch_multi_head_selection_re) and have been making significant modifications in this repository for improving the SOTA performance of Summarization models by introducing a `new idea` of incorporating learnings from Dependency Parser Networks along with word-level and NER embeddings using multiple Graphical Convolutional Neural Network (GCNConv) variants and use this as the encoder features which are to be fed to decoder network for performing Summarization.
The experiments are still a {{WORK-IN-PROGRESS}} and hopefully would be completed soon. 
I am currently pushing my changes to `feature/experiments` branch {{Please track this branch for latest commits}}. (Performed a master-merge recently).

This model will be extreamly useful for real-world usage.

I am mainly using CoNLL04 data for my training.

# Requirement

* python 3.7
* pytorch 1.10
* PyTorch Geometric (Latest Version)

# Datasets Available In This Repository

## CoNLL04

We use the data processed by official version.

**already in ./raw_data/CoNLL04/**


# Run
```shell
python main.py --mode preprocessing --exp_name conll_bert_re
python main.py --mode train --exp_name conll_bert_re 
python main.py --mode evaluation --exp_name conll_bert_re
```

# Current Work-In-Progress

* Experimenting with multiple Graphical Neural Network variants to create the best possible encoder encoding to be fed to decoder. 

## Citation

[paper](https://arxiv.org/abs/1804.07847)