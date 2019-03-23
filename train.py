#-*- coding: utf-8 -*-

import math
import os
import argparse
import json
import shutil
import pprint

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, TabularDataset, BucketIterator
from torchtext.vocab import FastText, GloVe
from tqdm import tqdm

from nets import Embedding, LSTMEncoder, NSE, LSTMDecoder, Seq2seq

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train', required=True, help='TSV file for training (.tsv)')
    parser.add_argument('--valid', required=True, help='TSV file for validation (.tsv)')
    parser.add_argument('--save-dir', required=True, help='Save directory')
    parser.add_argument('--encoder', default='LSTM', choices=['LSTM', 'BiLSTM', 'NSE'],
                        help='Select type of encoder')
    parser.add_argument('--attn', default=False, action='store_true',
                        help='Whether to add attention layer')
    parser.add_argument('--batchsize', type=int, default=64, help='Batch size')
    parser.add_argument('--embsize', type=int, default=128, help='Embedding size')
    parser.add_argument('--unit', type=int, default=128, help='Number of unit')
    parser.add_argument('--layer', type=int, default=2, help='Number of layer')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--clip', type=float, default=10, help='Clipping gradients')
    parser.add_argument('--epoch', type=int, default=25, help='Max epoch')
    parser.add_argument('--minfreq', type=int, default=2, help='Min word frequency')
    parser.add_argument('--vocabsize', type=int, default=40000, help='vocabulary size')
    parser.add_argument('--init-emb', choices=['fasttext', 'glove', 'none'], default='none',
                        help='Select pretrained word embeddings')
    parser.add_argument('--share-emb', default=False, action='store_true',
                        help='Whether to share embedding layers')
    args = parser.parse_args()

    ### setup data ###
    print('setup data...\n')
    SRC = Field(init_token='<sos>', eos_token='<eos>', lower=True)
    TRG = Field(init_token='<sos>', eos_token='<eos>', lower=True)

    train_data, valid_data = TabularDataset.splits(
        path='.',
        train=args.train,
        validation=args.valid,
        format='tsv',
        fields=[('src', SRC), ('trg', TRG)]
    )

    # pre-trained word embeddingsを使うかどうか
    if args.init_emb == 'none':
        SRC.build_vocab(train_data, min_freq=args.minfreq, max_size=args.vocabsize)
        TRG.build_vocab(train_data, min_freq=args.minfreq, max_size=args.vocabsize)
    else:
        if args.init_emb == 'fasttext':
            vectors = FastText(language='en')
        elif args.init_emb == 'glove':
            vectors = GloVe()
        SRC.build_vocab(train_data, vectors=vectors, min_freq=args.minfreq, max_size=args.vocabsize)
        TRG.build_vocab(train_data, vectors=vectors, min_freq=args.minfreq, max_size=args.vocabsize)
        args.embsize = SRC.vocab.vectors.size()[1]

    vocabs = {'src_stoi': SRC.vocab.stoi, 'src_itos': SRC.vocab.itos,
              'trg_stoi': TRG.vocab.stoi, 'trg_itos': TRG.vocab.itos}

    train_iter, valid_iter = BucketIterator.splits(
        (train_data, valid_data),
        batch_size=args.batchsize,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        repeat=False,
        shuffle=True,
        device=device
    )

    train_size = len(train_data)
    valid_size = len(valid_data)
    src_vocabsize = len(SRC.vocab)
    trg_vocabsize = len(TRG.vocab)
    print(f'# training examples: {train_size}')
    print(f'# validation examples: {valid_size} \n')
    print(f'# unique tokens in source vocabulary: {src_vocabsize}')
    print(f'# unique tokens in target vocabulary: {trg_vocabsize} \n')

    ### setup model ###
    sos_id = TRG.vocab.stoi['<sos>']
    # Embedding Layerを共有するかどうか
    if args.share_emb:
        max_vocabsize = max(src_vocabsize, trg_vocabsize)
        embedding = Embedding(max_vocabsize, args.embsize)
        encoder_embedding = embedding
        decoder_embedding = embedding
    else:
        encoder_embedding = Embedding(src_vocabsize, args.embsize)
        decoder_embedding = Embedding(trg_vocabsize, args.embsize)
    # Encoderでbidirectionalにするかどうか
    bidirectional = True if args.encoder == 'BiLSTM' else False

    if args.encoder == 'NSE':
        encoder = NSE(encoder_embedding, args.unit, args.layer, args.dropout)
    else:
        encoder = LSTMEncoder(encoder_embedding, args.unit, args.layer, args.dropout, bidirectional)
    decoder = LSTMDecoder(decoder_embedding, args.unit, args.layer,
                          args.dropout, args.attn, encoder.output_units)
    model = Seq2seq(encoder, decoder, sos_id, device).to(device)
    print(model)
    print()

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=TRG.vocab.stoi['<pad>'])

    ### make directory for saving ###
    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.mkdir(args.save_dir)

    ### save parameters ###
    params = args.__dict__
    params.update(train_size=train_size, valid_size=valid_size,
                  src_vocabsize=src_vocabsize, trg_vocabsize=trg_vocabsize)
    json.dump(params, open(args.save_dir + '/params.json', 'w', encoding='utf-8'), ensure_ascii=False)
    print('parameters:')
    pprint.pprint(params, indent=4)
    print()

    ### training and validation ###
    best_loss = float('inf')
    for epoch in range(args.epoch):
        train_loss = train(model, train_iter, optimizer, criterion, args.clip)
        valid_loss = eval(model, valid_iter, criterion)
        # save model
        model_path = args.save_dir + f'/model-e{epoch+1:02}.pt'
        state = {'vocabs': vocabs, 'params': params, 'state_dict': model.state_dict()}
        torch.save(state, model_path)

        # save best model
        if valid_loss < best_loss:
            best_loss = valid_loss
            model_path = args.save_dir + '/model-best.pt'
            torch.save(state, model_path)

        # logging
        logs = f"Epoch: {epoch+1:02}\tTrain loss: {train_loss:.3f}\tVal. Loss: {valid_loss:.3f}\n"
        print(logs)
        with open(args.save_dir + '/logs.txt', 'a') as f:
            f.write(logs)


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
            loss = criterion(outs[1:].view(-1, outs.shape[2]), trg[1:].view(-1))
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


if __name__ == '__main__':
    main()
