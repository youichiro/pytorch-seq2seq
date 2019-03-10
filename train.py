#-*- coding: utf-8 -*-

import math
import os
import argparse
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, TabularDataset, BucketIterator
from tqdm import tqdm

from nets import RNNEncoder, RNNAttnDecoder, Seq2seq

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train', required=True, help='TSV file for training (.tsv)')
    parser.add_argument('--valid', required=True, help='TSV file for validation (.tsv)')
    parser.add_argument('--save-dir', required=True, help='Save directory')
    # parser.add_argument('--model', required=True, help='Model name (.pt)')
    parser.add_argument('--attn', choices=['dot', 'general', 'concat'], default='dot',
                        help='Select attention method')
    parser.add_argument('--batchsize', type=int, default=64, help='Batch size')
    parser.add_argument('--embsize', type=int, default=128, help='Embedding size')
    parser.add_argument('--unit', type=int, default=128, help='Number of unit')
    parser.add_argument('--layer', type=int, default=2, help='Number of layer')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--clip', type=float, default=10, help='Clipping gradients')
    parser.add_argument('--epoch', type=int, default=25, help='Max epoch')
    parser.add_argument('--minfreq', type=int, default=2, help='Min word frequency')
    parser.add_argument('--vocabsize', type=int, default=40000, help='vocabulary size')
    args = parser.parse_args()

    # setup data
    SRC = Field(init_token='<sos>', eos_token='<eos>', lower=True)
    TRG = Field(init_token='<sos>', eos_token='<eos>', lower=True)

    train_data, valid_data = TabularDataset.splits(
        path='.',
        train=args.train,
        validation=args.valid,
        format='tsv',
        fields=[('src', SRC), ('trg', TRG)]
    )

    train_size = len(train_data)
    valid_size = len(valid_data)
    print(f'# training examples: {train_size}')
    print(f'# validation examples: {valid_size}')
    print('')

    SRC.build_vocab(train_data, min_freq=args.minfreq, max_size=args.vocabsize)
    TRG.build_vocab(train_data, min_freq=args.minfreq, max_size=args.vocabsize)

    vocabs = {'src_stoi': SRC.vocab.stoi, 'src_itos': SRC.vocab.itos,
              'trg_stoi': TRG.vocab.stoi, 'trg_itos': TRG.vocab.itos}

    src_vocabsize = len(SRC.vocab)
    trg_vocabsize = len(TRG.vocab)
    print(f'Unique tokens in source vocabulary: {src_vocabsize}')
    print(f'Unique tokens in target vocabulary: {trg_vocabsize}')
    print('')

    train_iter, valid_iter = BucketIterator.splits(
        (train_data, valid_data),
        batch_size=args.batchsize,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        repeat=False,
        device=device
    )

    # setup model
    sos_id = TRG.vocab.stoi['<sos>']
    encoder = RNNEncoder(src_vocabsize, args.embsize, args.unit, args.layer, args.dropout)
    decoder = RNNAttnDecoder(trg_vocabsize, args.embsize, args.unit, args.layer, args.dropout, args.attn)
    model = Seq2seq(encoder, decoder, sos_id, device).to(device)
    print(model)
    print()

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=TRG.vocab.stoi['<pad>'])

    # make directory for saving model
    os.makedirs()(args.save_dir, exist_ok=True)

    # save options
    params = args.__dict__
    params.update(train_size=train_size, valid_size=valid_size, src_vocabsize=src_vocabsize, trg_vocabsize=trg_vocabsize)
    json.dump(params, open(args.save_dir + '/params.json', 'w', encoding='utf-8'), ensure_ascii=False)

    # training and validation
    best_loss = float('inf')
    for epoch in range(args.epoch):
        train_loss = train(model, train_iter, optimizer, criterion, args.clip)
        valid_loss = eval(model, valid_iter, criterion)
        model_path = args.save_dir + f'/model-e{epoch+1:03}.pt'
        state = {'vocabs': vocabs, 'params': params, 'state_dict': model.state_dict()}
        torch.save(state, model_path)

        if valid_loss < best_loss:
            best_loss = valid_loss
            model_path = args.save_dir + '/model-best.pt'
            torch.save(state, model_path)


        logs = f"""
         | Epoch: {epoch+1:03}
         | Train loss: {train_loss:.3f}
         | Train PPL: {math.exp(train_loss):7.3f}
         | Val. Loss {valid_loss:.3f}
         | Val. PPL: {math.exp(valid_loss): 7.3f} |
        """

        print(logs)
        with open(args.save_dir + '/logs.txt', 'a') as f:
            f.write(logs)
        # print(f' | Epoch: {epoch+1:03}', end='')
        # print(f' | Train loss: {train_loss:.3f}', end='')
        # print(f' | Train PPL: {math.exp(train_loss):7.3f}', end='')
        # print(f' | Val. Loss {valid_loss:.3f}', end='')
        # print(f' | Val. PPL: {math.exp(valid_loss): 7.3f} |')


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for batch in tqdm(iterator):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        outs = model(src, trg, None)
        loss = criterion(outs[1:].view(-1, outs.shape[2]), trg[1:].view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def eval(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch in iterator:
            src = batch.src
            trg = batch.trg
            outs = model(src, trg, None, 0)
            loss = criterion(outs[1:].view(-1, outs.shape[2]),
                             trg[1:].view(-1))
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


if __name__ == '__main__':
    main()
