#-*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random


class Embedding(nn.Module):
    def __init__(self, n_vocab, n_emb):
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, n_emb)
        self.n_vocab = n_vocab
        self.n_emb = n_emb

    def forward(self, words):
        return self.embedding(words)


class RNNEncoder(nn.Module):
    def __init__(self, embedding, n_unit, n_layer, dropout, bidirectional=False):
        super().__init__()
        self.embedding = embedding
        self.rnn = nn.LSTM(embedding.n_emb, n_unit, n_layer, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_seqs):
        embedded = self.dropout(self.embedding(src_seqs))
        enc_outs, hs = self.rnn(embedded)
        return enc_outs, hs


class RNNDecoder(nn.Module):
    def __init__(self, embedding, n_unit, n_layer, dropout):
        super().__init__()
        self.embedding = embedding
        self.rnn = nn.LSTM(embedding.n_emb, n_unit, n_layer)
        self.wo = nn.Linear(n_unit, embedding.n_vocab)
        self.dropout = nn.Dropout(dropout)
        self.n_vocab = embedding.n_vocab

    def forward(self, tgt_words, hs):
        tgt_words = tgt_words.unsqueeze(0)  # 次元を一つ上げる
        embedded = self.dropout(self.embedding(tgt_words))
        dec_outs, hs = self.rnn(embedded, hs)
        pred = self.wo(dec_outs.squeeze(0))  # 次元を一つ下げる
        return pred, hs


class Attention(nn.Module):
    def __init__(self, method, n_unit):
        super().__init__()
        self.method = method
        self.n_unit = n_unit
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "Is not an appropriate attention method.")
        if method == 'general':
            self.w = nn.Linear(n_unit, n_unit)
        elif method == 'concat':
            self.w = nn.Linear(n_unit * 2, n_unit)
            self.v = torch.nn.Parameter(torch.FloatTensor(n_unit))

    def forward(self, dec_out, enc_outs):
        if self.method == 'dot':
            attn_energies = self.dot_score(dec_out, enc_outs)
        elif self.method == 'general':
            attn_energies = self.general_score(dec_out, enc_outs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(dec_out, enc_outs)
        return F.softmax(attn_energies, dim=0)

    def dot_score(self, dec_out, enc_outs):
        return torch.sum(dec_out * enc_outs, dim=2)

    def general_score(self, dec_out, enc_outs):
        energy = self.w(enc_outs)
        return torch.sum(dec_out * energy, dim=2)

    def concat_score(self, dec_out, enc_outs):
        dec_out = dec_out.expand(enc_outs.shape[0], -1, -1)
        energy = torch.cat((dec_out, enc_outs), 2)
        return torch.sum(self.v * self.w(energy).tanh(), dim=2)


class RNNAttnDecoder(nn.Module):
    def __init__(self, embedding, n_unit, n_layer, dropout, attn):
        super().__init__()
        self.embedding = embedding
        self.rnn = nn.LSTM(embedding.n_emb, n_unit, n_layer)
        self.attn = Attention(attn, n_unit)
        self.wc = nn.Linear(n_unit * 2, n_unit)
        self.wo = nn.Linear(n_unit, embedding.n_vocab)
        self.dropout = nn.Dropout(dropout)
        self.n_vocab = embedding.n_vocab

    def forward(self, tgt_words, hs, enc_outs):
        tgt_words = tgt_words.unsqueeze(0)
        embedded = self.dropout(self.embedding(tgt_words))
        dec_out, hs = self.rnn(embedded, hs)
        attn_weights = self.attn(dec_out, enc_outs)
        context = torch.bmm(attn_weights.transpose(1, 0).unsqueeze(1),
                            enc_outs.transpose(1, 0)
                  ).transpose(1, 0)
        cats = self.wc(torch.cat((dec_out, context), dim=2)).tanh()
        pred = self.wo(cats.squeeze(0))
        return pred, hs


class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder, sos_id, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sos_id = sos_id
        self.device = device

    def forward(self, src_seqs, tgt_seqs, maxlen, teaching_force_ratio=0.5):
        bs = src_seqs.shape[1]
        enc_outs, hs = self.encoder(src_seqs)

        if tgt_seqs is None:  # testing mode
            inputs = torch.ones(bs, dtype=torch.int64) * self.sos_id
            inputs = inputs.to(self.device)
        else:  # training mode
            maxlen = tgt_seqs.shape[0]
            inputs = tgt_seqs[0]

        outputs = torch.zeros((maxlen, bs, self.decoder.n_vocab))
        outputs = outputs.to(self.device)
        for i in range(1, maxlen):
            # biのときここでhsのback-wardだけ使うか，sumとるか，concatするか
            preds, hs = self.decoder(inputs, hs, enc_outs)
            outputs[i] = preds
            teaching_force = random.random() < teaching_force_ratio
            top1 = preds.max(1)[1]
            inputs = (tgt_seqs[i] if teaching_force else top1)
        return outputs
