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


class NSE(nn.Module):
    def __init__(self, embedding, n_unit, n_layer, dropout):
        super().__init__()
        self.embedding = embedding
        self.read_lstm = nn.LSTM(embedding.n_emb, n_unit, n_layer)
        self.compose_mlp = nn.Linear(2 * n_unit, 2 * n_unit)
        self.write_lstm = nn.LSTM(2 * n_unit, n_unit, n_layer)
        self.dropout = nn.Dropout(dropout)
        self.n_unit = n_unit
        self.n_layer = n_layer
        self.n_emb = embedding.n_emb
        self.output_units = n_unit
        assert self.n_emb == self.n_unit, "Should be same embsize and n_unit"

    def forward(self, src_seqs):
        srclen, batch = src_seqs.size()  # sec_seqs: (srclen, batch)
        embedd = self.dropout(self.embedding(src_seqs))  # embedd: (srclen, batch, n_unit)
        M_t = embedd.transpose(0, 1)  # M_t: (batch, srclen, n_unit)
        outputs = []
        for l in range(srclen):
            x_t = embedd[l, :, :]  # x_t: (batch, n_unit)
            o_t, m_t, z_t = self.read(M_t, x_t)
            c_t = self.compose(o_t, m_t)
            M_t, h_t, hs = self.write(M_t, c_t, z_t)
            outputs.append(h_t)
        outputs = torch.stack(outputs, dim=0)  # outputs: (srclen, batch, n_unit)
        return outputs, hs

    def read(self, M_t, x_t):
        # M_t: (batch, srclen, n_unit)
        # x_t: (batch, n_unit)
        batch, srclen, n_unit = M_t.size()
        o_t, _ = self.read_lstm(x_t.unsqueeze(0))  # o_t: (1, batch n_unit)
        o_t = o_t.view(batch, n_unit)  # o_t: (batch, n_unit)
        # o_t.unsqueeze(2): (batch, n_unit, 1)
        # torch.matmul(M_t, o_t.unsqueeze(2)): (batch, srclen, 1)
        z_t = F.softmax(torch.matmul(M_t, o_t.unsqueeze(2)).view(batch, srclen), dim=1)  # z_t: (batch, srclen)
        # z_t.unsqueeze(1): (batch, 1, srclen)
        m_t = torch.matmul(z_t.unsqueeze(1), M_t).view(batch, n_unit)  # m_t: (batch, n_unit)
        return o_t, m_t, z_t

    def compose(self, o_t, m_t):
        # o_t: (batch, n_unit)
        # m_t: (batch, n_unit)
        c_t = self.compose_mlp(torch.cat((o_t, m_t), dim=1))  # c_t: (batch, 2 * n_unit)
        return c_t


    def write(self, M_t, c_t, z_t):
        # M_t: (batch, srclen, n_unit)
        # c_t: (batch, 2 * n_unit)
        # z_t: (batch, srclen)
        batch, srclen, n_unit = M_t.size()
        h_t, hs = self.write_lstm(c_t.unsqueeze(0))  # h_t: (1, batch, n_unit)
        h_t = h_t.view(batch, n_unit)  # h_t: (batch, n_unit)
        # (1 - z_t).view(batch, srclen, 1): (batch, srclen, 1)
        # (1 - z_t).view(batch, srclen, 1).repeat(1, 1, n_unit): (batch, srclen, n_unit)
        M_t = (1 - z_t).view(batch, srclen, 1).repeat(1, 1, n_unit) * M_t  # M_t: (batch, srclen, n_unit)
        # z_t.view(batch, srclen, 1).repeat(1, 1, n_unit): (batch, srclen, n_unit)
        # h_t.view(batch, 1, n_unit).repeat(1, srclen, 1): (batch, srclen, n_unit)
        M_t += z_t.view(batch, srclen, 1).repeat(1, 1, n_unit) * h_t.view(batch, 1, n_unit).repeat(1, srclen, 1)  # M_t: (batch, srclen, n_unit)
        return M_t, h_t, hs


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
