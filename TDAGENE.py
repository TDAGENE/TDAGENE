import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optm
from torch.nn import CosineSimilarity
import networkx as nx
import numpy as np


def compute_tda_feature(x, adj, device, filtration_scale=1.5):

    cos = CosineSimilarity(dim=1)
    indices = adj.coalesce().indices()  
    mask = indices[0] < indices[1]  
    src = indices[0][mask]
    dst = indices[1][mask]
    s = cos(x[src], x[dst])
    filtration_values = (1 - s) * filtration_scale  
    edges = [(u.item(), v.item(), w.item()) for u, v, w in zip(src, dst, filtration_values)]
    n = x.size(0)
    if not edges:
        return torch.tensor([0.0, 0, 0.0, 0.0], dtype=torch.float, device=device)
    edges.sort(key=lambda t: t[2])
    parent = list(range(n))
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    persistence0 = []
    persistence1 = []
    for u, v, w in edges:
        pu, pv = find(u), find(v)
        if pu != pv:
            parent[pu] = pv
            persistence0.append((0, w))
        else:
            persistence1.append((w, float('inf')))
    if persistence0:
        pers0 = [d - b for b, d in persistence0]
        avg_pers0 = np.mean(pers0)
        max_pers0 = np.max(pers0)
        norm_pers0 = avg_pers0 / (max_pers0 + 1e-8)
    else:
        avg_pers0 = 0.0
        max_pers0 = 0.0
        norm_pers0 = 0.0
    num_cycles = len(persistence1)
    tda_feature = torch.tensor([avg_pers0, num_cycles, max_pers0, norm_pers0], dtype=torch.float, device=device)
    return tda_feature


class GENELink(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim, num_head1, num_head2,
                 alpha, device, type, reduction, dropout=0.1, decoder_type=None, fusion_type='gate', extra_feat_dim=0):
        super(GENELink, self).__init__()
        self.num_head1 = num_head1
        self.num_head2 = num_head2
        self.device = device
        self.alpha = alpha
        self.type = type
        self.reduction = reduction
        self.dropout = dropout
        self.fusion_type = fusion_type
        self.extra_feat_dim = extra_feat_dim

        self.tda_dim = 4

        if self.reduction == 'mean':
            self.hidden1_dim = hidden1_dim
            self.hidden2_dim = hidden2_dim
        elif self.reduction == 'concate':
            self.hidden1_dim = num_head1 * hidden1_dim
            self.hidden2_dim = num_head2 * hidden2_dim
        else:
            raise ValueError("Invalid reduction type. Choose 'mean' or 'concate'.")

        self.effective_input_dim = input_dim + extra_feat_dim

        self.ConvLayer1 = nn.ModuleList([AttentionLayer(self.effective_input_dim, hidden1_dim, alpha) for _ in range(num_head1)])
        self.ConvLayer2 = nn.ModuleList([AttentionLayer(self.hidden1_dim, hidden2_dim, alpha) for _ in range(num_head2)])
        self.ConvLayer3 = nn.ModuleList([AttentionLayer(self.hidden2_dim, hidden2_dim, alpha) for _ in range(num_head2)])
        self.ConvLayer4 = nn.ModuleList([AttentionLayer(self.hidden2_dim, hidden2_dim, alpha) for _ in range(num_head2)])

        self.proj1 = nn.Linear(self.effective_input_dim, self.hidden1_dim) if self.effective_input_dim != self.hidden1_dim else nn.Identity()
        self.proj2 = nn.Linear(self.hidden1_dim, self.hidden2_dim) if self.hidden1_dim != self.hidden2_dim else nn.Identity()
        self.proj3 = nn.Linear(self.hidden2_dim, self.hidden2_dim) if self.hidden2_dim != self.hidden2_dim else nn.Identity()
        self.proj4 = nn.Linear(self.hidden2_dim, self.hidden2_dim) if self.hidden2_dim != self.hidden2_dim else nn.Identity()

        self.film_gamma = nn.Linear(self.tda_dim, self.hidden2_dim)
        self.film_beta = nn.Linear(self.tda_dim, self.hidden2_dim)

        self.gate_linear = nn.Linear(self.hidden2_dim + self.tda_dim, self.hidden2_dim + self.tda_dim)
        self.gate_extra_linear = nn.Linear(self.hidden2_dim + self.tda_dim, self.hidden2_dim + self.tda_dim)
        self.gate_adapter = nn.Linear(self.hidden2_dim + self.tda_dim, self.hidden2_dim)

        self.additive_proj = nn.Linear(self.tda_dim, self.hidden2_dim)

        self.fusion_proj = nn.Sequential(
            nn.Linear(self.hidden2_dim, self.hidden2_dim),
            nn.LeakyReLU(self.alpha),
            nn.Linear(self.hidden2_dim, self.hidden2_dim)
        )

        self.tf_linear1 = nn.Linear(self.hidden2_dim, hidden3_dim)
        self.tf_linear2 = nn.Linear(hidden3_dim, hidden3_dim)
        self.tf_linear3 = nn.Linear(hidden3_dim, hidden3_dim)
        self.tf_ln1 = nn.LayerNorm(hidden3_dim)
        self.tf_ln2 = nn.LayerNorm(hidden3_dim)
        self.tf_ln3 = nn.LayerNorm(hidden3_dim)
        self.tf_final = nn.Linear(hidden3_dim, output_dim)

        self.target_linear1 = nn.Linear(self.hidden2_dim, hidden3_dim)
        self.target_linear2 = nn.Linear(hidden3_dim, hidden3_dim)
        self.target_linear3 = nn.Linear(hidden3_dim, hidden3_dim)
        self.target_ln1 = nn.LayerNorm(hidden3_dim)
        self.target_ln2 = nn.LayerNorm(hidden3_dim)
        self.target_ln3 = nn.LayerNorm(hidden3_dim)
        self.target_final = nn.Linear(hidden3_dim, output_dim)

        self.decoder_type = decoder_type or self.type
        if self.decoder_type == 'MLP':
            self.linear = nn.Sequential(
                nn.Linear(2 * output_dim, max(64, output_dim)),
                nn.LayerNorm(max(64, output_dim)),
                nn.LeakyReLU(self.alpha),
                nn.Dropout(self.dropout),
                nn.Linear(max(64, output_dim), max(32, output_dim // 2)),
                nn.LayerNorm(max(32, output_dim // 2)),
                nn.LeakyReLU(self.alpha),
                nn.Dropout(self.dropout),
                nn.Linear(max(32, output_dim // 2), 1)
            )
        elif self.decoder_type == 'bilinear':
            self.bilinear = nn.Bilinear(output_dim, output_dim, 1, bias=False)
        elif self.decoder_type in ['dot', 'cosine']:
            pass
        else:
            raise ValueError(f"Invalid decode type: {self.decoder_type}. Choose 'dot', 'cosine', 'MLP', or 'bilinear'.")

        self.reset_parameters()

    def reset_parameters(self):
        for attention in self.ConvLayer1:
            attention.reset_parameters()
        for attention in self.ConvLayer2:
            attention.reset_parameters()
        for attention in self.ConvLayer3:
            attention.reset_parameters()
        for attention in self.ConvLayer4:
            attention.reset_parameters()

        nn.init.kaiming_uniform_(self.tf_linear2.weight, a=self.alpha, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.tf_linear3.weight, a=self.alpha, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.tf_final.weight, a=self.alpha, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.target_linear2.weight, a=self.alpha, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.target_linear3.weight, a=self.alpha, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.target_final.weight, a=self.alpha, nonlinearity='leaky_relu')

        nn.init.kaiming_uniform_(self.tf_linear1.weight, a=self.alpha, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.target_linear1.weight, a=self.alpha, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.gate_linear.weight, a=self.alpha, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.gate_extra_linear.weight, a=self.alpha, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.gate_adapter.weight, a=self.alpha, nonlinearity='leaky_relu')
        for layer in self.fusion_proj:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=self.alpha, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.additive_proj.weight, a=self.alpha, nonlinearity='leaky_relu')
        nn.init.ones_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

        if hasattr(self, 'bilinear'):
            nn.init.xavier_uniform_(self.bilinear.weight)

    def encode(self, x, adj):
        degrees = torch.sparse.sum(adj, dim=1).to_dense().unsqueeze(1)
        degrees = F.normalize(degrees, p=2, dim=0)
        if self.extra_feat_dim > 0:
            x = torch.cat([x, degrees], dim=1)

        previous = self.proj1(x)

        outs1 = [att(x, adj) for att in self.ConvLayer1]
        if self.reduction == 'concate':
            out = torch.cat(outs1, dim=1)
        elif self.reduction == 'mean':
            out = torch.mean(torch.stack(outs1), dim=0)
        out = F.elu(out + previous)
        previous = self.proj2(out)

        outs2 = [att(out, adj) for att in self.ConvLayer2]
        if self.reduction == 'concate':
            out = torch.cat(outs2, dim=1)
        elif self.reduction == 'mean':
            out = torch.mean(torch.stack(outs2), dim=0)
        out = F.elu(out + previous)
        previous = self.proj3(out)

        outs3 = [att(out, adj) for att in self.ConvLayer3]
        if self.reduction == 'concate':
            out = torch.cat(outs3, dim=1)
        elif self.reduction == 'mean':
            out = torch.mean(torch.stack(outs3), dim=0)
        out = F.elu(out + previous)
        previous = self.proj4(out)

        outs4 = [att(out, adj) for att in self.ConvLayer4]
        if self.reduction == 'concate':
            out = torch.cat(outs4, dim=1)
        elif self.reduction == 'mean':
            out = torch.mean(torch.stack(outs4), dim=0)
        out = F.elu(out + previous)

        return out

    def _film_fuse(self, gat_embed, tda_feature):
        gamma = torch.sigmoid(self.film_gamma(tda_feature))
        beta = self.film_beta(tda_feature)
        fused = gamma * gat_embed + beta
        proj = self.fusion_proj(fused)
        fused = F.leaky_relu(proj + gat_embed, negative_slope=self.alpha)
        return fused

    def _gate_fuse(self, gat_embed, tda_feature):
        concat = torch.cat([gat_embed, tda_feature], dim=1)
        gate = torch.sigmoid(self.gate_linear(concat))
        gated = gate * concat
        gated = F.leaky_relu(gated, negative_slope=self.alpha)
        gated = self.gate_extra_linear(gated)
        fused = self.gate_adapter(gated)
        fused = F.leaky_relu(fused, negative_slope=self.alpha)
        return fused

    def _additive_fuse(self, gat_embed, tda_feature):
        tda_proj = self.additive_proj(tda_feature)
        fused = gat_embed + tda_proj
        fused = F.leaky_relu(fused, negative_slope=self.alpha)
        return fused

    def decode(self, tf_embed, target_embed):
        if self.decoder_type == 'dot':
            logits = torch.mul(tf_embed, target_embed)
            logits = torch.sum(logits, dim=1).view(-1, 1)
            return logits
        elif self.decoder_type == 'cosine':
            sim = torch.cosine_similarity(tf_embed, target_embed, dim=1).view(-1, 1)
            clamped = sim.clamp(min=-0.999, max=0.999)
            logits = 0.5 * torch.log1p(clamped) - 0.5 * torch.log1p(-clamped)
            return logits
        elif self.decoder_type == 'MLP':
            h = torch.cat([tf_embed, target_embed], dim=1)
            logits = self.linear(h)
            return logits
        elif self.decoder_type == 'bilinear':
            logits = self.bilinear(tf_embed, target_embed)
            return logits

    def compute_embeddings(self, x, adj, tda_feature):
        gat_embed = self.encode(x, adj)
        tda_stream = tda_feature.unsqueeze(0).repeat(x.size(0), 1)
        if self.fusion_type == 'film':
            fused_embed = self._film_fuse(gat_embed, tda_stream)
        elif self.fusion_type == 'gate':
            fused_embed = self._gate_fuse(gat_embed, tda_stream)
        elif self.fusion_type == 'additive':
            fused_embed = self._additive_fuse(gat_embed, tda_stream)
        else:
            raise ValueError("fusion_type must be 'gate', 'film', or 'additive'")

        tf_embed = self.tf_linear1(fused_embed)
        tf_embed = self.tf_ln1(tf_embed)
        tf_embed = F.leaky_relu(tf_embed, negative_slope=self.alpha)
        tf_embed = F.dropout(tf_embed, p=self.dropout, training=self.training)
        
        tf_embed = self.tf_linear2(tf_embed)
        tf_embed = self.tf_ln2(tf_embed)
        tf_embed = F.leaky_relu(tf_embed, negative_slope=self.alpha)
        tf_embed = F.dropout(tf_embed, p=self.dropout, training=self.training)
        
        tf_embed = self.tf_linear3(tf_embed)
        tf_embed = self.tf_ln3(tf_embed)
        tf_embed = F.leaky_relu(tf_embed, negative_slope=self.alpha)
        tf_embed = F.dropout(tf_embed, p=self.dropout, training=self.training)
        
        tf_embed = self.tf_final(tf_embed)
        tf_embed = F.leaky_relu(tf_embed, negative_slope=self.alpha)

        target_embed = self.target_linear1(fused_embed)
        target_embed = self.target_ln1(target_embed)
        target_embed = F.leaky_relu(target_embed, negative_slope=self.alpha)
        target_embed = F.dropout(target_embed, p=self.dropout, training=self.training)
        
        target_embed = self.target_linear2(target_embed)
        target_embed = self.target_ln2(target_embed)
        target_embed = F.leaky_relu(target_embed, negative_slope=self.alpha)
        target_embed = F.dropout(target_embed, p=self.dropout, training=self.training)
        
        target_embed = self.target_linear3(target_embed)
        target_embed = self.target_ln3(target_embed)
        target_embed = F.leaky_relu(target_embed, negative_slope=self.alpha)
        target_embed = F.dropout(target_embed, p=self.dropout, training=self.training)
        
        target_embed = self.target_final(target_embed)
        target_embed = F.leaky_relu(target_embed, negative_slope=self.alpha)

        self.tf_output = tf_embed
        self.target_output = target_embed
        return tf_embed, target_embed

    def forward(self, x, adj, train_sample, tda_feature):
        tf_embed, target_embed = self.compute_embeddings(x, adj, tda_feature)
        train_tf = tf_embed[train_sample[:, 0]]
        train_target = target_embed[train_sample[:, 1]]
        pred_logit = self.decode(train_tf, train_target)
        return pred_logit

    def get_embedding(self):
        return self.tf_output, self.target_output


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=0.2, bias=True):
        super(AttentionLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha

        self.weight = nn.Parameter(torch.FloatTensor(self.input_dim, self.output_dim))
        self.a = nn.Parameter(torch.zeros(size=(2 * self.output_dim, 1)))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.output_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight.data, a=self.alpha, nonlinearity='leaky_relu')
        if self.bias is not None:
            self.bias.data.fill_(0)
        nn.init.kaiming_uniform_(self.a.data, a=self.alpha, nonlinearity='leaky_relu')

    def _prepare_attentional_mechanism_input(self, x):
        Wh1 = torch.matmul(x, self.a[:self.output_dim, :])
        Wh2 = torch.matmul(x, self.a[self.output_dim:, :])
        e = F.leaky_relu(Wh1 + Wh2.T, negative_slope=self.alpha)
        return e

    def forward(self, x, adj):
        h = torch.matmul(x, self.weight)
        e = self._prepare_attentional_mechanism_input(h)

        adj_indices = adj.coalesce().indices()
        adj_values = adj.coalesce().values()
        sparse_e = torch.sparse.FloatTensor(adj_indices, e[adj_indices[0], adj_indices[1]], adj.size())

        attention = F.softmax(sparse_e.to_dense(), dim=1)

        attention = F.dropout(attention, training=self.training)
        h_pass = torch.matmul(attention, h)

        output_data = h_pass
        output_data = F.leaky_relu(output_data, negative_slope=self.alpha)
        output_data = F.normalize(output_data, p=2, dim=1)

        if self.bias is not None:
            output_data = output_data + self.bias

        return output_data