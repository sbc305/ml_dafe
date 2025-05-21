from model import subsequent_mask

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

def generate_summary(model, source_inputs, source_mask, sos_idx, eos_idx, max_len=50):
    model.eval()
    with torch.no_grad():
        batch_size = source_inputs.size(0)
        generated = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=source_inputs.device)

        for _ in range(max_len):
            tgt_mask = (generated != pad_idx).unsqueeze(1) & subsequent_mask(generated.size(1)).type_as(source_mask)
            out = model(source_inputs, generated, source_mask, tgt_mask)
            next_token_logits = out[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
            if (next_token == eos_idx).all():
                break
        return generated[0].tolist()

def generate_summary(model, source_inputs, source_mask, sos_idx, eos_idx, max_len=50, k=10):
    batch_size = source_inputs.size(0)
    generated = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=source_inputs.device)

    for _ in range(max_len):
        target_mask = subsequent_mask(generated.size(1)).to(source_inputs.device)
        logits = model(source_inputs, generated, source_mask, target_mask)
        next_token_logits = logits[:, -1, :]

        topk_logits, topk_indices = torch.topk(next_token_logits, k)
        probs = torch.nn.functional.softmax(topk_logits, dim=-1)
        next_token = topk_indices.gather(1, torch.multinomial(probs, 1))

        generated = torch.cat([generated, next_token], dim=1)

        if (next_token == eos_idx).all():
            break

    return generated[0].tolist()

def beam_search(model, source_inputs, source_mask, sos_idx, eos_idx, beam_width=3, max_len=50):
    device = source_inputs.device
    batch_size = source_inputs.size(0)
    
    sequences = [[list(), 0.0]]
    
    for _ in range(max_len):
        all_candidates = []
        for seq, score in sequences:
            if len(seq) > 0 and seq[-1] == eos_idx:
                all_candidates.append((seq, score))
                continue
            
            inp_seq = torch.tensor([ [sos_idx] + seq ], device=device)
            inp_mask = subsequent_mask(inp_seq.size(1)).to(device)
            logits = model(source_inputs, inp_seq, source_mask, inp_mask)
            logits = logits[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
            
            top_log_probs, top_indices = torch.topk(log_probs, beam_width)
            
            for i in range(beam_width):
                candidate_seq = seq + [top_indices[i].item()]
                candidate_score = score - top_log_probs[i].item()
                all_candidates.append((candidate_seq, candidate_score))
        
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        sequences = ordered[:beam_width]
        
        if all(seq[-1] == eos_idx for seq, _ in sequences):
            break
    
    return sequences[0][0]

def generate_summary_with_temperature(model, source_inputs, source_mask, sos_idx, eos_idx, max_len=50, temperature=1.2):
    generated = torch.full((1,1), sos_idx, dtype=torch.long, device=source_inputs.device)
    for _ in range(max_len):
        target_mask = subsequent_mask(generated.size(1)).to(source_inputs.device)
        logits = model(source_inputs, generated, source_mask, target_mask)
        logits = logits[:, -1, :] / temperature
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        generated = torch.cat([generated, next_token], dim=1)
        if next_token.item() == eos_idx:
            break
    return generated[0].tolist()
