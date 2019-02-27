# -*- coding: utf-8 -*-

import argparse
import torch
from nets import RNNEncoder, RNNAttnDecoder, Seq2seq

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(file_path, stoi):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            words = [stoi[token.lower()] for token in line.rstrip().split(' ')]
            data.append(torch.LongTensor(words))
    return data

def get_sentence(sequence, itos):
    sentence = [itos[i] for i in sequence]
    if '<eos>' in sentence:
        return sentence[:sentence.index('<eos>')]
    return sentence


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', required=True, help='Model file (.pt)')
    parser.add_argument('--test', required=True, help='Test file')
    parser.add_argument('--maxlen', type=int, default=20, help='Max length')
    args = parser.parse_args()

    # load model state
    state = torch.load(args.model)
    params = state['params']
    src_stoi = state['vocabs']['src_stoi']
    src_itos = state['vocabs']['src_itos']
    trg_stoi = state['vocabs']['trg_stoi']
    trg_itos = state['vocabs']['trg_itos']

    sos_id = trg_stoi['<sos>']
    encoder = RNNEncoder(len(src_itos), params['n_emb'], params['n_unit'],
                         params['n_layer'], 0.0)
    decoder = RNNAttnDecoder(len(trg_itos), params['n_emb'], params['n_unit'],
                             params['n_layer'], 0.0, params['attn'])
    model = Seq2seq(encoder, decoder, sos_id, device).to(device)
    model.load_state_dict(state['state_dict'])

    # load source file
    src_data = load_data(args.test, src_stoi)

    for sequence in src_data:
        sequence = sequence.view(sequence.shape[0], -1).to(device)
        output = model(sequence, None, args.maxlen, 0.0)
        output = output.squeeze(1)[1:].max(1)[1]
        output = get_sentence(output, trg_itos)
        print(' '.join(output))


if __name__ == '__main__':
    main()
