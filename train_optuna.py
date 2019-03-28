#-*- coding: utf-8 -*-

import os
import argparse
import json
import shutil
import pprint
import subprocess
import optuna

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
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
    parser.add_argument('--attn', default=False, action='store_true', help='Whether to add attention layer')
    parser.add_argument('--batchsize', type=int, default=64, help='Batch size')
    parser.add_argument('--embsize', type=int, default=128, help='Embedding size')
    parser.add_argument('--unit', type=int, default=128, help='Number of unit')
    parser.add_argument('--clip', type=float, default=10, help='Clipping gradients')
    parser.add_argument('--epoch', type=int, default=25, help='Max epoch')
    parser.add_argument('--minfreq', type=int, default=2, help='Min word frequency')
    parser.add_argument('--maxlen', type=int, default=70, help='Max number of words for validation')
    parser.add_argument('--early-stop-n', type=int, default=2,
                        help='Stop training if the best score does not update  n epoch before')
    parser.add_argument('--n-trial', type=int, default=100, help='Number of trial')
    parser.add_argument('--study-name', default=None, help='Study name for sqlite')
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
        fields=[('src', SRC), ('trg', TRG)])


    def objective(trial):
        ### setup trials ###
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
        lr_schedule_gamma = trial.suggest_loguniform('lr-schedule-gamma', 0.5, 1.0)
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
        layer = trial.suggest_categorical('n_layer', [1, 2, 3])
        dropout = trial.suggest_uniform('dropout', 0.0, 0.3)
        vocabsize = trial.suggest_int('vocabsize', 20000, 40000)
        init_emb = trial.suggest_categorical('init-emb', ['fasttext', 'glove', 'none'])
        share_emb = trial.suggest_categorical('share-emb', [True, False])

        ### setup vocabularies ###
        if init_emb == 'none':
            SRC.build_vocab(train_data, min_freq=args.minfreq, max_size=vocabsize)
            TRG.build_vocab(train_data, min_freq=args.minfreq, max_size=vocabsize)
        else:
            if init_emb == 'fasttext':
                vectors = FastText(language='en')
            elif init_emb == 'glove':
                vectors = GloVe()
            SRC.build_vocab(train_data, vectors=vectors, min_freq=args.minfreq, max_size=vocabsize)
            TRG.build_vocab(train_data, vectors=vectors, min_freq=args.minfreq, max_size=vocabsize)
            args.embsize = SRC.vocab.vectors.size()[1]

        vocabs = {'src_stoi': SRC.vocab.stoi, 'src_itos': SRC.vocab.itos,
                  'trg_stoi': TRG.vocab.stoi, 'trg_itos': TRG.vocab.itos }

        train_iter, valid_iter = BucketIterator.splits(
            (train_data, valid_data),
            batch_size=args.batchsize,
            sort_within_batch=True,
            sort_key=lambda x: len(x.src),
            repeat=False,
            shuffle=True,
            device=device)

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
        eos_id = TRG.vocab.stoi['<eos>']
        src_pad_id = SRC.vocab.stoi['<pad>']
        trg_pad_id = TRG.vocab.stoi['<pad>']

        if share_emb:
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
            encoder = NSE(encoder_embedding, args.unit, layer, dropout)
        else:
            encoder = LSTMEncoder(encoder_embedding, args.unit, layer, dropout, bidirectional)

        decoder = LSTMDecoder(decoder_embedding, args.unit, layer, dropout, args.attn, encoder.output_units)
        model = Seq2seq(encoder, decoder, sos_id, eos_id, device).to(device)
        parameter_num = count_parameters(model)
        print(model)

        # Multi GPU
        if device.__str__() == 'cuda':
            model = torch.nn.DataParallel(model)
            cudnn.benchmark = True

        criterion = nn.CrossEntropyLoss(ignore_index=TRG.vocab.stoi['<pad>'])
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # lr scheduling with exponential curve
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=lr_schedule_gamma)

        ### make directory for saving ###
        if os.path.exists(args.save_dir):
            shutil.rmtree(args.save_dir)
        os.mkdir(args.save_dir)

        params = args.__dict__
        params.update(lr=lr, weight_decay=weight_decay, layer=layer, dropout=dropout, vocabsize=vocabsize,
                      init_emb=init_emb, share_emb=share_emb, train_size=train_size, valid_size=valid_size,
                      src_vocabsize=src_vocabsize, trg_vocabsize=trg_vocabsize, parameter_num=parameter_num)
        json.dump(params, open(f'{args.save_dir}/params.json', 'w', encoding='utf-8'), ensure_ascii=False)
        pprint.pprint(params, indent=4)
        print()

        ### training and validation ###
        best_loss = float('inf')
        no_update_best_interval = 0
        for epoch in range(args.epoch):
            is_first = True if epoch == 0 else False
            scheduler.step()  # reduce lr
            train_loss = train(model, train_iter, optimizer, criterion, args.clip)
            valid_loss = eval(model, valid_iter, criterion, SRC.vocab.itos, TRG.vocab.itos, is_first)

            if valid_loss < best_loss:
                best_loss = valid_loss
                no_update_best_interval = 0
                # save best model
                model_path = f'{args.save_dir}/model-best.pt'
                state = {'vocabs': vocabs, 'params': params, 'state_dict': model.state_dict()}
                torch.save(state, model_path)
            else:
                no_update_best_interval += 1

            # logging
            logs = f"Epoch: {epoch+1:02}\tTrain loss: {train_loss:.3f}\tVal. Loss: {valid_loss:.3f}\n"
            print(logs)

            # early stopping
            if no_update_best_interval >= args.early_stop_n:
                print('Early stopped in training')
                break

        return best_loss

    if args.study_name:
        # You have to do the following command in advance:
        # optuna create-study --study '{args.study_name}' --storage 'sqlite:///{args.study_name}.db'
        study = optuna.Study(study_name=args.study_name, storage=f'sqlite:///{args.study_name}.db')
    else:
        study = optuna.create_study()
    study.optimize(objective, n_trials=args.n_trial)
    print('\nbest params: ', study.best_params)
    print(f'best value: {study.best_value}')
    print(f'best trial: {study.best_trial.trial_id}')


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

    with torch.no_grad():
        for batch in iterator:
            src = batch.src
            trg = batch.trg
            outs = model(src, trg, None, 0)
            loss = criterion(outs[1:].view(-1, outs.shape[2]),
                             trg[1:].view(-1))
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    main()
