from prepare_datasets import get_ds_iters
from general import *
from model import EncoderDecoder, NoamOpt, convert_batch
from LabelSmoothing import LabelSmoothingLoss
from generators import generate_summary

import pandas as pd
from tqdm.auto import tqdm
import os
import math
import wandb
from rouge_score import rouge_scorer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchtext.data import Field, Example, Dataset, BucketIterator

def do_epoch(model, criterion, data_iter, optimizer=None, name=None, save_path=None, compute_rouge=False, breakpoint_condition=None):
    epoch_loss = 0
    rouge1_total, rouge2_total, rouge_count = 0.0, 0.0, 0
    is_train = optimizer is not None
    name = name or ''
    model.train(is_train)

    batches_count = len(data_iter)

    with torch.autograd.set_grad_enabled(is_train):
        with tqdm(total=batches_count) as progress_bar:
            for i, batch in enumerate(data_iter):
                source_inputs, target_inputs, source_mask, target_mask = convert_batch(batch)
                logits = model.forward(source_inputs, target_inputs[:, :-1], source_mask, target_mask[:, :-1, :-1])

                logits = logits.contiguous().view(-1, logits.shape[-1])
                target = target_inputs[:, 1:].contiguous().view(-1)
                loss = criterion(logits, target)
                
                
                epoch_loss += loss.item()
                if compute_rouge:
                    sos_idx = word_field.vocab.stoi['<s>']
                    eos_idx = word_field.vocab.stoi['</s>']
                    pad_idx = word_field.vocab.stoi['<pad>']
                    predicted_ids = generate_summary(model, source_inputs, source_mask, sos_idx, eos_idx)
                    pred_words = [word_field.vocab.itos[i] for i in predicted_ids if i not in {sos_idx, eos_idx, pad_idx}]
                    pred_text = ' '.join(pred_words)
        
                    gold_ids = target_inputs[0].tolist()
                    gold_words = [word_field.vocab.itos[i] for i in gold_ids if i not in {sos_idx, eos_idx, pad_idx}]
                    gold_text = ' '.join(gold_words)
                    rouge_scores = scorer.score(gold_text, pred_text)
                    rouge1_total += rouge_scores['rouge1'].fmeasure
                    rouge2_total += rouge_scores['rouge2'].fmeasure
                    rouge_count += 1

                if breakpoint_condition and breakpoint_condition(loss.item(), i):
                    print(f"Breakpoint hit at batch {i}, loss = {loss.item():.5f}")
                    breakpoint()

                if optimizer:
                    optimizer.optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                progress_bar.update()
                progress_bar.set_description('{:>5s} Loss = {:.5f}, PPX = {:.2f}'.format(
                    name, loss.item(), math.exp(loss.item()))
                )

            avg_loss = epoch_loss / batches_count
            avg_rouge1 = rouge1_total / rouge_count if rouge_count else 0
            avg_rouge2 = rouge2_total / rouge_count if rouge_count else 0
            progress_bar.set_description('{:>5s} Loss = {:.5f}, PPX = {:.2f}, ROUGE-1 = {:.5f}, ROUGE-2 = {:.5f}'.format(
                name, avg_loss, math.exp(avg_loss), avg_rouge1, avg_rouge2)
            )
            progress_bar.refresh()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        wandb.save(save_path)
        print(f"Model saved to {save_path}")    

    return avg_loss, avg_rouge1, avg_rouge2

def fit(model, criterion, optimizer, train_iter, epochs_count=1, val_iter=None):
    best_val_loss = None
    
    train_losses = []
    val_losses = []
    rouge1_scores = []
    rouge2_scores = []

    wandb.init(project="hw3", name="model", config={"epochs": epochs_count})
    
    for epoch in range(epochs_count):
        name_prefix = '[{} / {}] '.format(epoch + 1, epochs_count)
        train_loss = do_epoch(model, criterion, train_iter, optimizer, name_prefix + 'Train:', save_path="./checkpoints/model{}.pth".format(epoch + 1))
        train_losses.append(train_loss)
        
        if not val_iter is None:
            val_loss, rouge1, rouge2 = do_epoch(model, criterion, val_iter, None, 
                                                name_prefix + '  Val:',
                                                compute_rouge=True,
                                                save_path=f"./checkpoints/model{epoch + 1}.pth")
            val_losses.append(val_loss)
            rouge1_scores.append(rouge1)
            rouge2_scores.append(rouge2)

        log_dict = {
            "Train Loss": train_loss[0],
            "Train PPX": math.exp(train_loss[0]),
            "Val Rouge1 Score": train_loss[1],
            "Val Rougt2 Score": train_loss[2],
            "Epoch": epoch + 1,
        }
            
        if val_iter is not None:
            log_dict.update({
                "Val Loss": val_loss,
                "Val PPX": math.exp(val_loss),
                "Val Rouge1 Score": rouge1,
                "Val Rougt2 Score": rouge2
            })
            
        wandb.log(log_dict)

    print (train_losses, val_losses, rouge1_scores, rouge2_scores)
    return train_losses, val_losses, rouge1_scores, rouge2_scores




scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)

d_model = 300
train_iter, test_iter = get_ds_iters("runtime/news.csv", d_model)
pad_idx = word_field.vocab.stoi['<pad>']
model = EncoderDecoder(source_vocab_size=len(word_field.vocab), target_vocab_size=len(word_field.vocab), d_model=d_model, pretrained_vectors=word_field.vocab.vectors).to(DEVICE)
criterion = LabelSmoothingLoss(0.1, len(word_field.vocab), ignore_index=pad_idx).to(DEVICE)
optimizer = NoamOpt(model)

os.makedirs(os.path.dirname("model/"), exist_ok=True)
torch.save(word_field.vocab, "model/vocab.pt")



train_losses, val_losses, rouge1_scores, rouge2_scores = fit(model, criterion, optimizer, train_iter, epochs_count=30, val_iter=test_iter)
