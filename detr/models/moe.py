from .transformer import TransformerDecoderLayer, TransformerEncoderLayer, TransformerEncoder, TransformerDecoder

import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

import IPython
e = IPython.embed

class TransformerMoE(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, num_experts=4, top_k=2,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, aux_loss= False):
        super().__init__()

        encoder_layer = TransformerEncoderLayerWithMoE(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, num_experts, top_k)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoderMoE(encoder_layer, num_encoder_layers, encoder_norm, aux_loss)

        decoder_layer = TransformerDecoderLayerWithMoE(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, num_experts, top_k)
        decoder_norm = nn.RMSNorm(d_model)
        self.decoder = TransformerDecoderMoE(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec, return_aux=aux_loss)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.return_aux = aux_loss

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, latent_input=None, proprio_input=None, additional_pos_embed=None):
        # TODO flatten only when input has H and W
        if len(src.shape) == 4: # has H and W
            # flatten NxCxHxW to HWxNxC
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1).repeat(1, bs, 1)
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
            # mask = mask.flatten(1)

            additional_pos_embed = additional_pos_embed.unsqueeze(1).repeat(1, bs, 1) # seq, bs, dim
            pos_embed = torch.cat([additional_pos_embed, pos_embed], axis=0)

            addition_input = torch.stack([latent_input, proprio_input], axis=0)
            src = torch.cat([addition_input, src], axis=0)
        else:
            assert len(src.shape) == 3
            # flatten NxHWxC to HWxNxC
            bs, hw, c = src.shape
            src = src.permute(1, 0, 2)
            pos_embed = pos_embed.unsqueeze(1).repeat(1, bs, 1)
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        tgt = torch.zeros_like(query_embed)
        memory, encoder_aux_loss = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs, decoder_aux_loss = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        hs = hs.transpose(1, 2)

        return hs, encoder_aux_loss, decoder_aux_loss


class Expert(nn.Module):
    def __init__(self, d_model, dim_feedforward):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, x):
        return self.ffn(x)

class MoELayer(nn.Module):
    def __init__(self, d_model, dim_feedforward, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert(d_model, dim_feedforward) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x, return_aux = False):
        # x: [seq_len, batch, dim]
        seq_len, batch_size, dim = x.size()
        x_flat = x.contiguous().view(-1, dim)  # [seq_len * batch, dim]

        scores = self.gate(x_flat)  # [seq_len * batch, num_experts]
        topk_vals, topk_idx = torch.topk(scores, self.top_k, dim=-1)  # [seq_len * batch, k]
        topk_probs = torch.softmax(topk_vals, dim=-1)  # [seq_len * batch, k]

        expert_outputs = torch.zeros_like(x_flat)

        for i in range(self.top_k):
            expert_idx = topk_idx[:, i]
            expert_prob = topk_probs[:, i].unsqueeze(-1)

            for e_idx in range(self.num_experts):
                mask = (expert_idx == e_idx)
                if mask.sum() == 0:
                    continue
                selected_inputs = x_flat[mask]
                outputs = self.experts[e_idx](selected_inputs)
                expert_outputs[mask] += expert_prob[mask] * outputs

        if return_aux:
            return expert_outputs.view(seq_len, batch_size, dim), topk_probs, topk_idx

        return expert_outputs.view(seq_len, batch_size, dim)
    
class TransformerEncoderLayerWithMoE(TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, num_experts=4, top_k=2):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)

        self.num_experts = num_experts
        self.top_k = top_k
        self.moe_layer = MoELayer(d_model, dim_feedforward, self.num_experts, self.top_k)
        # self.norm1 = nn.RMSNorm(d_model)
        # self.norm2 = nn.RMSNorm(d_model)

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None, return_aux = False):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        if return_aux:
            src2, topk_probs, topk_idx = self.moe_layer(src, return_aux = return_aux) 
            src = src + self.dropout2(src2)
            src = self.norm2(src)
            
            return src, topk_probs, topk_idx
        
        else:
            src2 = self.moe_layer(src) 
            src = src + self.dropout2(src2)
            src = self.norm2(src)

            return src, None, None

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    return_aux: Optional[bool] = False):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        if return_aux:
            src2, topk_probs, topk_idx = self.moe_layer(src2, return_aux)
            src = src + self.dropout2(src2)
        
            return src, topk_probs, topk_idx
        
        else:
            src2 = self.moe_layer(src2)
            src = src + self.dropout2(src2)
        
            return src, None, None

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                return_aux: Optional[bool] = False):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos, return_aux)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos, return_aux)

class TransformerDecoderLayerWithMoE(TransformerDecoderLayer):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, num_experts = 4, top_k = 2):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)

        self.num_experts = num_experts
        self.top_k = top_k
        
        self.moe_layer = MoELayer(d_model, dim_feedforward, self.num_experts, self.top_k)

        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.norm3 = nn.RMSNorm(d_model)
 

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     return_aux: Optional[bool] = False):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        if return_aux is False:
            tgt2 = self.moe_layer(tgt)
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)
            return tgt, None, None
        
        else:
            tgt2, topk_probs, topk_idx = self.moe_layer(tgt, return_aux= return_aux)
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)
            return tgt, topk_probs, topk_idx

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    return_aux: Optional[bool] = False):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)

        if return_aux:
            tgt2, topk_probs, topk_idx = self.moe_layer(tgt2)
            tgt = tgt + self.dropout3(tgt2)
            return tgt, topk_probs, topk_idx
        else:

            tgt2 = self.moe_layer(tgt2)
            tgt = tgt + self.dropout3(tgt2)
            return tgt, None, None

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                return_aux: Optional[bool] = False):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, return_aux)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, return_aux)


class TransformerEncoderMoE(nn.Module):

    def __init__(self, encoder_layer: TransformerEncoderLayerWithMoE, num_layers, norm=None, return_aux = False):
        super().__init__()

        self.return_aux = return_aux
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        all_aux_loss = None

        for layer in self.layers:
            output, topk_probs, topk_idx  = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos, return_aux = self.return_aux)

            if self.return_aux:
                if all_aux_loss is None:
                    all_aux_loss = (1/self.num_layers) * moe_aux_loss(topk_probs, topk_idx, layer.num_experts)
                else:
                    aux_loss_layer = moe_aux_loss(topk_probs, topk_idx, layer.num_experts)
                    all_aux_loss += (1/self.num_layers) * aux_loss_layer

        if self.norm is not None:
            output = self.norm(output)
        
        return output, all_aux_loss
    

class TransformerDecoderMoE(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, return_aux = False):
        super().__init__()
        self.return_aux = return_aux
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        all_aux_loss = None

        for layer in self.layers:
            output, topk_probs, topk_idx = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, return_aux = self.return_aux)
            
            if self.return_aux:
                if all_aux_loss is None:
                    all_aux_loss = (1/self.num_layers) * moe_aux_loss(topk_probs, topk_idx, layer.num_experts)
                else:
                    aux_loss_layer = moe_aux_loss(topk_probs, topk_idx, layer.num_experts)
                    all_aux_loss += (1/self.num_layers) * aux_loss_layer
            
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), all_aux_loss

        return output.unsqueeze(0), all_aux_loss


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def moe_aux_loss(gate_outputs, topk_idx, num_experts):
    """
    gate_outputs: softmax(topk_vals), shape [batch_size * seq_len, top_k]
    topk_idx: indices of top-k experts, shape [batch_size * seq_len, top_k]
    """
    # Count how many tokens are routed to each expert
    expert_usage = torch.zeros(num_experts, device=topk_idx.device)
    for i in range(topk_idx.size(1)):  # loop over top-k
        expert_ids = topk_idx[:, i]
        expert_usage.scatter_add_(0, expert_ids, gate_outputs[:, i])

    # Normalize
    expert_prob = expert_usage / expert_usage.sum()
    loss = (expert_prob * torch.log(expert_prob + 1e-9)).sum()  # entropy
    return -loss


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer_moe(args):
    return TransformerMoE(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        num_experts=args.num_experts,
        top_k=args.top_k,
        return_intermediate_dec=True,
        aux_loss = args.aux_loss
    )
