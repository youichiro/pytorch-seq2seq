# -*- coding: utf-8 -*-

import argparse
import torch
from nets import EmbeddingLayer, LSTMEncoder, NSE, LSTMDecoder, Seq2seq

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
    parser.add_argument('--beamsize', type=int, default=None, help='Num of Beam Search width')
    args = parser.parse_args()

    # load model state
    state = torch.load(args.model)
    params = state['params']
    src_stoi = state['vocabs']['src_stoi']
    src_itos = state['vocabs']['src_itos']
    trg_stoi = state['vocabs']['trg_stoi']
    trg_itos = state['vocabs']['trg_itos']
    src_pad_id = src_stoi['<pad>']
    trg_pad_id = trg_stoi['<pad>']

    sos_id = trg_stoi['<sos>']
    eos_id = trg_stoi['<eos>']
    if params['share_emb']:
        max_vocabsize = max(params['src_vocabsize'], params['trg_vocabsize'])
        assert src_pad_id == trg_pad_id
        embedding = EmbeddingLayer(max_vocabsize, params['embsize'], src_pad_id)
        encoder_embedding = embedding
        decoder_embedding = embedding
    else:
        encoder_embedding = EmbeddingLayer(params['src_vocabsize'], params['embsize'], src_pad_id)
        decoder_embedding = EmbeddingLayer(params['trg_vocabsize'], params['embsize'], trg_pad_id)
    bidirectional = True if params['encoder'] == 'BiLSTM' else False

    if params['encoder'] == 'NSE':
        encoder = NSE(encoder_embedding, params['unit'], params['layer'], 0.0)
    else:
        encoder = LSTMEncoder(encoder_embedding, params['unit'], params['layer'], 0.0, bidirectional)

    decoder = LSTMDecoder(decoder_embedding, params['unit'], params['layer'],
                          0.0, params['attn'], encoder.output_units)
    model = Seq2seq(encoder, decoder, sos_id, eos_id, device).to(device)
    model.load_state_dict(state['state_dict'])

    # load source file
    src_data = load_data(args.test, src_stoi)

    for sequence in src_data:
        sequence = sequence.view(sequence.shape[0], -1).to(device)
        if args.beamsize is None:
            output = model(sequence, None, args.maxlen, 0.0)  # output: (# sentence, 1, trglen)
            output = output.squeeze(1)[1:].max(1)[1]
        else:
            output = model.beam(sequence, None, args.maxlen, args.beamsize, topk=1)
            output = output[0][0][1:]

        output = get_sentence(output, trg_itos)
        if not output:
            print("NOTOKENS")
        else:
            print(' '.join(output))


if __name__ == '__main__':
    main()
