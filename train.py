#-*- coding: utf-8 -*-

import math
import os
import argparse
import json
import shutil
import pprint
import subprocess

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchtext.data import Field, TabularDataset, BucketIterator
from torchtext.vocab import FastText, GloVe
from tqdm import tqdm

from nets import EmbeddingLayer, LSTMEncoder, NSE, LSTMDecoder, Seq2seq
from translate import get_sentence

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
    src_pad_id = SRC.vocab.stoi['<pad>']
    trg_pad_id = TRG.vocab.stoi['<pad>']
    # Embedding Layerを共有するかどうか
    if args.share_emb:
        vocabsize = max(src_vocabsize, trg_vocabsize)
        assert src_pad_id == trg_pad_id
        embedding = EmbeddingLayer(vocabsize, args.embsize, src_pad_id)
        encoder_embedding = embedding
        decoder_embedding = embedding
    else:
        encoder_embedding = EmbeddingLayer(src_vocabsize, args.embsize, src_pad_id)
        decoder_embedding = EmbeddingLayer(trg_vocabsize, args.embsize, trg_pad_id)
    bidirectional = True if args.encoder == 'BiLSTM' else False

    if args.encoder == 'NSE':
        encoder = NSE(encoder_embedding, args.unit, args.layer, args.dropout)
    else:
        encoder = LSTMEncoder(encoder_embedding, args.unit, args.layer, args.dropout, bidirectional)

    decoder = LSTMDecoder(decoder_embedding, args.unit, args.layer,
                          args.dropout, args.attn, encoder.output_units)
    model = Seq2seq(encoder, decoder, sos_id, device).to(device)
    parameter_num = count_parameters(model)
    print(model)
    print(f'\n# parameters: {parameter_num}')
    print()

    # Multi GPU
    if device.__str__() == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=TRG.vocab.stoi['<pad>'])

    ### make directory for saving ###
    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.mkdir(args.save_dir)
    os.mkdir(args.save_dir + '/results')

    ### save parameters ###
    params = args.__dict__
    params.update(train_size=train_size, valid_size=valid_size,
                  src_vocabsize=src_vocabsize, trg_vocabsize=trg_vocabsize, parameter_num=parameter_num)
    json.dump(params, open(args.save_dir + '/params.json', 'w', encoding='utf-8'), ensure_ascii=False)
    print('parameters:')
    pprint.pprint(params, indent=4)
    print()

    ### training and validation ###
    best_loss = float('inf')
    for epoch in range(args.epoch):
        is_first = True if epoch == 0 else False
        train_loss = train(model, train_iter, optimizer, criterion, args.clip)
        valid_loss, sequences = eval(model, valid_iter, criterion, SRC.vocab.itos, TRG.vocab.itos, is_first)

        # save model
        model_path = f'{args.save_dir}/model-e{epoch+1:02}.pt'
        state = {'vocabs': vocabs, 'params': params, 'state_dict': model.state_dict()}
        torch.save(state, model_path)

        # save best model
        if valid_loss < best_loss:
            best_loss = valid_loss
            model_path = args.save_dir + '/model-best.pt'
            torch.save(state, model_path)

        # save validation src and trg if first epoch
        result_dir = f'{args.save_dir}/results'
        if is_first:
            valid_src_path = result_dir + '/valid.src'
            valid_tgt_path = result_dir + '/valid.tgt'
            valid_ref_m2_path = result_dir + '/valid.ref.m2'
            with open(valid_src_path, 'w') as f:
                for s in sequences[1]:
                    f.write(s + '\n')
            with open(valid_tgt_path, 'w') as f:
                for s in sequences[2]:
                    f.write(s + '\n')
            cmd = f'sh ./parallel_to_ref_m2.sh {valid_src_path} {valid_tgt_path} {valid_ref_m2_path}'
            subprocess.check_call(cmd.split())
            print(f'Create {valid_src_path}, {valid_tgt_path}, {valid_ref_m2_path}')

        # save validation outputs
        prefix = f'valid-e{epoch+1:02}'
        valid_output_path = f'{result_dir}/{prefix}.sys'
        with open(valid_output_path, 'w') as f:
            for output in sequences[0]:
                f.write(output + '\n')

        # compute ERRANT score
        cmd = f'sh ./compute_errant_score.sh {valid_output_path} {valid_tgt_path} {valid_ref_m2_path} {result_dir} {prefix}'
        errant_score = subprocess.check_output(cmd.split()).decode("UTF-8").strip()

        # logging
        logs = f"Epoch: {epoch+1:02}\tTrain loss: {train_loss:.3f}\tVal. Loss: {valid_loss:.3f}\tERRANT score: {errant_score}\n"
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


def eval(model, iterator, criterion, src_itos, trg_itos, is_first):
    model.eval()
    epoch_loss = 0
    outputs = []
    src_sentences = []
    trg_sentences = []

    with torch.no_grad():
        for batch in iterator:
            src = batch.src
            trg = batch.trg
            outs = model(src, trg, None, 0)
            loss = criterion(outs[1:].view(-1, outs.shape[2]), trg[1:].view(-1))
            epoch_loss += loss.item()
            # system output sentences
            batchsize = outs.size(1)
            for row in range(batchsize):
                s = outs[:, row, :][1:].max(1)[1]
                s = ' '.join(get_sentence(s, trg_itos))
                outputs.append(s)

            # validation sentences
            if is_first:
                src = src.transpose(0, 1)
                trg = trg.transpose(0, 1)
                for row in range(batchsize):
                    s = ' '.join(get_sentence(src[row][1:], src_itos))
                    t = ' '.join(get_sentence(trg[row][1:], trg_itos))
                    src_sentences.append(s)
                    trg_sentences.append(t)
            else:
                src_sentences = trg_sentences = None

    return epoch_loss / len(iterator), (outputs, src_sentences, trg_sentences)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    main()
