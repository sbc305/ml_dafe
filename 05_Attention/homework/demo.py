from torchtext.data import Iterator
from prepare_datasets import get_ds_iters
from general import *
from model import EncoderDecoder, convert_batch, subsequent_mask
from generators import *


import pandas as pd
from tqdm.auto import tqdm
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchtext.data import Field, Example, Dataset, BucketIterator
import matplotlib.pyplot as plt
import seaborn as sns

def show_attention(attn, source_tokens, target_tokens, title, path="./", head=0):
    attn = attn[0, head].detach().cpu()
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn, xticklabels=source_tokens, yticklabels=target_tokens, cmap="viridis")
    plt.xlabel("Source")
    plt.ylabel("Target")
    plt.title(f"{title}\nAttention Head {head}")
    plt.savefig(path)

my_test = [
    ("", "Заходит скелет в бар и говорит: мне пиво и швабру.", "Анекдот про скелет в баре"),
    ("", "Штирлиц и Мюллер стреляли по очереди. Очередь редела, но не расходилась", "Анекдот про Штирлица и очередь"),
    ("", "Штирлиц шел с трудом разбирая дорогу. К утру было разобрано 10 км железнодорожных путей.", "Анекдот про Штирлица и железную дорогу."),
    ("", "Заходит лошадь в бар, а бармен ей и говорит: слышь, а че морда такая длинная?", "Анекдот про лошадь в баре."),
    ("", "Штирлиц наблюдал лыжников в фуфайках, которые куда-то шли. Фуфлыжники - ловко догадался Штирлиц.", "Анекдот про Штирлица и фуфлыжников"),
]



df = pd.read_csv("runtime/news.csv", delimiter=',')
sample = df.sample(n=5, random_state=42)
my_df = pd.DataFrame(my_test, columns=df.columns)
df = pd.concat([sample, my_df], ignore_index=True)

fields = [('source', word_field), ('target', word_field)]
examples = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    source_text = word_field.preprocess(row.text)
    target_text = word_field.preprocess(row.title)
    examples.append(Example.fromlist([source_text, target_text], fields))

dataset = Dataset(examples, fields)
iterator = Iterator(dataset, batch_size=1, sort=False, device=DEVICE)


word_field.vocab = torch.load("vocab.pt")
sos_idx = word_field.vocab.stoi[BOS_TOKEN]
eos_idx = word_field.vocab.stoi[EOS_TOKEN]
pad_idx = word_field.vocab.stoi[PAD_TOKEN]


model = EncoderDecoder(source_vocab_size=len(word_field.vocab), target_vocab_size=len(word_field.vocab)).to(DEVICE)
model.load_state_dict(torch.load("model25.pth", weights_only=True, map_location=DEVICE))
model.eval()
print("Model loaded successfully.")


from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
rouge1_total, rouge2_total = 0.0, 0.0


with open("output.txt", "w") as f:
    for i, batch in enumerate(iterator):
        source_inputs, target_inputs, source_mask, target_mask = convert_batch(batch)
        summary_ids = generate_summary_with_temperature(model, source_inputs, source_mask, sos_idx, eos_idx)

        summary_words = [
            word_field.vocab.itos[i]
            for i in summary_ids
            if i not in {sos_idx, eos_idx, pad_idx, word_field.vocab.stoi['<unk>']}
        ]
        summary = ' '.join(summary_words)

        gold_ids = target_inputs[0].tolist()
        gold_words = [word_field.vocab.itos[i] for i in gold_ids if i not in {sos_idx, eos_idx, pad_idx}]
        gold_text = ' '.join(gold_words)

        scores = scorer.score(gold_text, summary)
        rouge1_total += scores['rouge1'].fmeasure
        rouge2_total += scores['rouge2'].fmeasure


        f.write(f"{gold_text};{summary}\n")

        attn = model.decoder._blocks[0]._encoder_attn._attn_probs
        show_attention(attn, source_inputs, target_inputs, title=gold_text, path=f"./attention/{i}.png")


print(f"ROUGE-1: {rouge1_total/len(examples) :.4f}")
print(f"ROUGE-2: {rouge2_total/len(examples) :.4f}")
