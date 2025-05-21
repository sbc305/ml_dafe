import pandas as pd
from tqdm.auto import tqdm
import os

import torch
from torchtext.data import Field, Example, Dataset, BucketIterator
from torchtext.vocab import GloVe

from general import *

def get_ds_iters(path="news.csv", dim=256):
    fields = [('source', word_field), ('target', word_field)]


    data = pd.read_csv(path, delimiter=',')

    examples = []
    for _, row in tqdm(data.iterrows(), total=len(data)):
        source_text = word_field.preprocess(row.text)
        target_text = word_field.preprocess(row.title)
        examples.append(Example.fromlist([source_text, target_text], fields))

    dataset = Dataset(examples, fields)
    train_dataset, test_dataset = dataset.split(split_ratio=0.85)


    word_field.build_vocab(train_dataset, vectors=GloVe(name='6B', dim=dim, cache='.vector_cache'))

    print('Train size =', len(train_dataset))
    print('Test size =', len(test_dataset))
    print('Vocab size =', len(word_field.vocab))

    return BucketIterator.splits(datasets=(train_dataset, test_dataset), batch_sizes=(16, 32), shuffle=True, device=DEVICE, sort=False)
