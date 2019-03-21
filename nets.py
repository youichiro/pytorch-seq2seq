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


class LSTMEncoder(nn.Module):
    def __init__(self, embedding, n_unit, n_layer, dropout, bidirectional=False):
        super().__init__()
        self.embedding = embedding
        self.rnn = nn.LSTM(embedding.n_emb, n_unit, n_layer, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.n_layer = n_layer
        self.bidirectional = bidirectional
        self.output_units = n_unit
        if bidirectional:
            self.output_units *= 2

    def forward(self, src_seqs):
        batch = src_seqs.size(1)
        embedded = self.dropout(self.embedding(src_seqs))
        enc_outs, (hs, cs) = self.rnn(embedded)  # hs: (n_layer * n_directions, batch, n_unit)

        if self.bidirectional:
            def combine_bidir(outs):
                return outs.view(self.n_layer, 2, batch, -1).transpose(1, 2).contiguous().view(self.n_layer, batch, -1)
            hs = combine_bidir(hs)
            cs = combine_bidir(cs)

        return enc_outs, (hs, cs)


class AttentionLayer(nn.Module):
    def __init__(self, dec_embed_dim, enc_embed_dim):
        super().__init__()
        self.input_proj = nn.Linear(dec_embed_dim, enc_embed_dim)

    def forward(self, dec_out, enc_outs):
        dec_out = self.input_proj(dec_out)
        attn_scores = (enc_outs * dec_out).sum(dim=2)
        attn_scores = F.softmax(attn_scores, dim=0)
        return attn_scores


class LSTMDecoder(nn.Module):
    def __init__(self, embedding, n_unit, n_layer, dropout, attention, encoder_output_units):
        super().__init__()
        self.embedding = embedding
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(embedding.n_emb, n_unit, n_layer)
        self.wo = nn.Linear(n_unit, embedding.n_vocab)
        self.encoder_output_units = encoder_output_units
        self.n_vocab = embedding.n_vocab
        if attention:
            self.attn = AttentionLayer(n_unit, encoder_output_units)
            self.wc = nn.Linear(n_unit + encoder_output_units, n_unit)
        else:
            self.attn = None
        if encoder_output_units != n_unit:
            self.encoder_hidden_proj = nn.Linear(encoder_output_units, n_unit)
            self.encoder_cell_proj = nn.Linear(encoder_output_units, n_unit)
        else:
            self.encoder_hidden_proj = self.encoder_cell_proj = None

    def forward(self, tgt_words, enc_outs, hs):
        tgt_words = tgt_words.unsqueeze(0)
        embedded = self.dropout(self.embedding(tgt_words))
        hiddens = hs[0]
        cells = hs[1]
        if self.encoder_hidden_proj is not None and hiddens.size(2) == self.encoder_output_units:
            hiddens = self.encoder_hidden_proj(hiddens)
            cells = self.encoder_cell_proj(cells)
        dec_out, hs = self.rnn(embedded, (hiddens, cells))
        if self.attn:
            attn_weights = self.attn(dec_out, enc_outs)
            context = torch.bmm(attn_weights.transpose(1, 0).unsqueeze(1),
                                enc_outs.transpose(1, 0)).transpose(1, 0)
            cats = self.wc(torch.cat((dec_out, context), dim=2)).tanh()
        else:
            cats = dec_out
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
            preds, hs = self.decoder(inputs, enc_outs, hs)
            outputs[i] = preds
            teaching_force = random.random() < teaching_force_ratio
            top1 = preds.max(1)[1]
            inputs = (tgt_seqs[i] if teaching_force else top1)
        return outputs
