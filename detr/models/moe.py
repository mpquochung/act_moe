from .transformer import TransformerDecoderLayer, TransformerEncoderLayer, TransformerEncoder, TransformerDecoder

import copy
from typing import Optional, Union, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

import IPython
e = IPython.embed

class TransformerMoE(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, num_experts=4, top_k=2,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayerWithMoE(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, num_experts, top_k)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayerWithMoE(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, num_experts, top_k)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

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
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        hs = hs.transpose(1, 2)
        return hs

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

    def forward(self, x):
        # x: [seq_len, batch, dim]
        seq_len, batch_size, dim = x.size()
        x_flat = x.view(-1, dim)  # [seq_len * batch, dim]

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

        return expert_outputs.view(seq_len, batch_size, dim)
    
    
class MoELayerLoadBalance(nn.Module):
    def __init__(self, d_model, dim_feedforward, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert(d_model, dim_feedforward) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)
        self.expert_biases = nn.Parameter(torch.zeros(num_experts))

    def forward(self, x):
        # x: [seq_len, batch, dim]
        seq_len, batch_size, dim = x.size()
        x_flat = x.view(-1, dim)  # [seq_len * batch, dim]

        scores = self.gate(x_flat)  # [seq_len * batch, num_experts]
        gate_probs = torch.sigmoid(scores)
        gate_logits = scores + self.expert_biases
        _, topk_idx = torch.topk(gate_logits, self.top_k, dim=-1)  # [seq_len * batch, k]
        top_k_probs = gate_probs.gather(-1, topk_idx)  # [seq_len * batch, k]
        
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        expert_outputs = torch.zeros_like(x_flat)

        for i in range(self.top_k):
            expert_idx = topk_idx[:, i]
            expert_prob = top_k_probs[:, i].unsqueeze(-1)

            for e_idx in range(self.num_experts):
                mask = (expert_idx == e_idx)
                if mask.sum() == 0:
                    continue
                selected_inputs = x_flat[mask]
                outputs = self.experts[e_idx](selected_inputs)
                expert_outputs[mask] += expert_prob[mask] * outputs

        return expert_outputs.view(seq_len, batch_size, dim)


class TransformerEncoderLayerWithMoE(TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, num_experts=4, top_k=2):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)

        # Replace FFN with MoE
        self.moe_layer = MoELayerLoadBalance(d_model, dim_feedforward, num_experts, top_k)
        # self.norm1 = nn.RMSNorm(d_model)
        # self.norm2 = nn.RMSNorm(d_model)

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.moe_layer(src)  # Replaces FFN with MoE

        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)

        src2 = self.moe_layer(src2)

        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerDecoderLayerWithMoE(TransformerDecoderLayer):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, num_experts = 4, top_k = 2):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
       
        
        self.moe_layer = MoELayerLoadBalance(d_model, dim_feedforward, num_experts, top_k)

        # self.norm1 = nn.RMSNorm(d_model)
        # self.norm2 = nn.RMSNorm(d_model)
        # self.norm3 = nn.RMSNorm(d_model)


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
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
        tgt2 = self.moe_layer(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
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
        tgt2 = self.moe_layer(tgt2)
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, Tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://huggingface.co/papers/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


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
    )
