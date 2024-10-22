# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import warnings
from typing import List, Optional, Tuple, Union
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from torch.utils.checkpoint import checkpoint

from .configuration_ref_adapter import REFAdapterConfig

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, dropout=0.1, max_len=256, batch_first=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        if not batch_first:
            pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        if self.batch_first:
            x = x + self.pe[:, :x.size(1), :]
        else:
            x = x + self.pe[:x.size(0), :, :]
        return self.dropout(x)

class REFAdapterEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(REFAdapterEncoderLayer, self).__init__()
        
        # Self-attention layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

        # Layer norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        # Self-attention block
        residual = x
        x = self.norm1(x)
        attn_outputs = self.self_attn(x, x, x, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        x = residual + self.dropout1(attn_outputs)

        # Feedforward block
        residual = x
        x = self.norm2(x)
        ffn_output = self.ffn(x)
        x = residual + self.dropout2(ffn_output)        
        return x

class REFAdapterEncoder(PreTrainedModel):

    def __init__(self, config: REFAdapterConfig):
        super().__init__(config)
        self.config = config
        v_input = config.v_input
        d_input = config.d_input
        d_model = config.d_model
        n_heads = config.n_heads
        dropout = config.dropout
        d_ffn = config.d_ffn
        max_position_embedding = config.max_position_embedding

        if d_input != d_model:
            self.in_proj = nn.Linear(d_input, d_model)
        if v_input != d_model:
            self.visual_in_proj = nn.Linear(v_input, d_model)
        self.positional_encoding = PositionalEncoding(
            d_model, 
            dropout=0, 
            max_len=max_position_embedding,
            batch_first=True
        )

        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            self.layers.append(
                REFAdapterEncoderLayer(
                    d_model,
                    n_heads,
                    d_ffn,
                    dropout
                )
            )
        self.last_norm = nn.LayerNorm(d_model)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(self,
        hidden_states,
        visual_hidden_states=None,
        attention_mask=None, 
        mode='loss'
    ):
        if self.config.d_input != self.config.d_model:
            hidden_states = self.in_proj(hidden_states)
        if visual_hidden_states is not None:
            if isinstance(visual_hidden_states,List):
                visual_hidden_states = visual_hidden_states[-1]
            if self.config.v_input != self.config.d_model:
                visual_hidden_states = self.visual_in_proj(visual_hidden_states)
            hidden_states = torch.cat([hidden_states,visual_hidden_states],dim=1)
        hidden_states = self.positional_encoding(hidden_states)

        all_hidden_states = ()
        for layer in self.layers:
            all_hidden_states += (hidden_states,)
            hidden_states = layer(
                hidden_states,
                src_mask=attention_mask
            )

        # add hidden states from the last decoder layer
        last_hidden_states = self.last_norm(hidden_states)
        all_hidden_states += (last_hidden_states,)
        
        return dict(
            last_hidden_states = last_hidden_states,
            hidden_states = all_hidden_states
        )

class REFAdapterDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(REFAdapterDecoderLayer, self).__init__()
        
        # Self-attention layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Cross-attention layer (attends to encoder output)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

        # Layer norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        
        # Self-attention block
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention block
        tgt2 = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feedforward block
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class REFAdapterDecoder(PreTrainedModel):

    def __init__(self, config: REFAdapterConfig):
        super().__init__(config)
        self.config = config
        num_queries_text = config.phrase_max_length + config.unit_max_length
        num_queries_ref = config.ref_max_length
        d_input = config.d_input
        d_model = config.d_model
        n_heads = config.n_heads
        dropout = config.dropout
        d_ffn = config.d_ffn

        if d_input != d_model:
            self.in_proj = nn.Linear(d_input, d_model)
        self.text_positional_encoding = PositionalEncoding(
            d_model, 
            dropout=0, 
            max_len=num_queries_text,
            batch_first=True
        )
        self.ref_positional_encoding = PositionalEncoding(
            d_model, 
            dropout=0, 
            max_len=num_queries_ref,
            batch_first=True
        )
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            self.layers.append(
                REFAdapterDecoderLayer(
                    d_model,
                    n_heads,
                    d_ffn,
                    dropout
                )
            )
        self.last_norm = nn.LayerNorm(d_model)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(self,
        ref_hidden_states,
        text_hidden_states, 
        attention_mask=None, 
        mode='loss'
    ):
        if self.config.d_input != self.config.d_model:
            ref_hidden_states = self.in_proj(ref_hidden_states)
            text_hidden_states = self.in_proj(text_hidden_states)
        
        ref_hidden_states = self.ref_positional_encoding(ref_hidden_states)
        text_hidden_states = self.text_positional_encoding(text_hidden_states)

        hidden_states = ref_hidden_states
        all_hidden_states = ()
        for layer in self.layers:
            all_hidden_states += (hidden_states,)
            hidden_states = layer(
                tgt=hidden_states,
                memory=text_hidden_states,
                tgt_mask=attention_mask
            )

        # add hidden states from the last decoder layer
        last_hidden_states = self.last_norm(hidden_states)
        all_hidden_states += (last_hidden_states,)
        
        return dict(
            last_hidden_states = last_hidden_states,
            hidden_states = all_hidden_states
        )


class REFAdapterModel(PreTrainedModel):
    _auto_class = 'AutoModel'
    config_class = REFAdapterConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True

    def __init__(self, config: REFAdapterConfig):
        super().__init__(config)
        self.config = config
        if 'encode' in config.mode:
            self.model = REFAdapterEncoder(config)
        elif config.mode == 'decode':
            if config.packing:
                self.register_buffer("split_ids", torch.arange(2).unsqueeze(1), persistent=False)
                self.split_embedding = nn.Embedding(2, config.d_input)
            self.model = REFAdapterDecoder(config)
        else:
            assert config.mode == 'projector'
            self.model = nn.Linear(config.d_input, config.d_model)
        self.post_init()

    def enable_input_require_grads(self):
        
        def make_inputs_require_grad(module, input, output):
            if isinstance(output, dict):
                for key in output.keys():
                    if isinstance(output[key], torch.Tensor):
                        output[key].requires_grad_(True)
            else:
                output.requires_grad_(True)
        
        self.model.register_forward_hook(make_inputs_require_grad)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, REFAdapterModel):
            module.gradient_checkpointing = value

    def padding_sequences(self, sequences, target_length):
        padded_sequences = []
        for seqs in sequences:
            for feats in seqs:
                length, dim = feats.shape
                if length > target_length:
                    padded_feats = feats[:target_length, :]
                else:
                    padded_feats = torch.zeros(
                        (target_length, dim), 
                        device=feats.device,
                        dtype=feats.dtype
                    )
                    padded_feats[:length, :] = feats
                padded_sequences.append(padded_feats)
        padded_sequences = torch.stack(padded_sequences, dim=0)
        return padded_sequences
    
    def reform_padded_sequences(self, ref_sequences, hidden_states):
        batch_size = len(ref_sequences)
        _, ref_pad_len, ref_dim = hidden_states.shape
        counter = torch.zeros((batch_size, 2), dtype=torch.uint8)

        for batch_idx, seqs in enumerate(ref_sequences):
            num_seqs = len(seqs)
            num_token = sum([feats.shape[0] for feats in seqs])
            counter[batch_idx, 0] = num_seqs
            counter[batch_idx, 1] = num_token

        max_ref_num = counter[:, 1].max().item()
        new_hidden_states = torch.zeros(
            (batch_size, max_ref_num, ref_dim), 
            dtype=hidden_states.dtype, 
            device=hidden_states.device
        )
        hidden_states_mask = torch.zeros(
            (batch_size, max_ref_num), 
            dtype=torch.bool, 
            device=hidden_states.device
        )
        start = 0
        for batch_idx, batch_counter in enumerate(counter):
            seqs = ref_sequences[batch_idx]
            num_seqs = batch_counter[0]
            num_token = batch_counter[1]

            end = start + num_seqs
            ref_mask = torch.zeros((num_seqs, ref_pad_len), dtype=torch.bool)
            for idx, feats in enumerate(seqs):
                ref_len = feats.shape[0]
                ref_mask[idx, :ref_len] = True
            
            num_ref_in_batch = ref_mask.sum()
            new_hidden_states[batch_idx, :num_ref_in_batch, :] = hidden_states[start:end][ref_mask]
            hidden_states_mask[batch_idx, :num_ref_in_batch] = True
        
        return new_hidden_states, hidden_states_mask

    def packing_sequences(self, sequences):
        start_embedding, end_embedding = self.split_embedding(self.split_ids)
        packed_sequences = []
        packed_masks = []
        max_feats_len = 0
        for seqs in sequences:
            new_feats = []
            new_masks = []
            for feats in seqs:
                feats = torch.cat([start_embedding, feats, end_embedding], dim=0)
                mask = torch.zeros((feats.shape[0]), device=feats.device, dtype=torch.bool)
                mask[1:-1] = True
                new_feats.append(feats)
                new_masks.append(mask)
            new_feats = torch.cat(new_feats, dim=0) # [L, C]
            new_masks = torch.cat(new_masks, dim=0) # [L]
            max_feats_len = max(new_feats.shape[0], max_feats_len)
            packed_sequences.append(new_feats)
            packed_masks.append(new_masks)

        padded_packed_sequences = []
        padded_packed_masks = []
        for feats, masks in zip(packed_sequences, packed_masks):
            padded_feats = torch.zeros(
                (max_feats_len, feats.shape[1]),
                device=feats.device,
                dtype=feats.dtype 
            )
            padded_masks = torch.zeros(
                (max_feats_len),
                device=masks.device,
                dtype=masks.dtype 
            )
            padded_feats[:feats.shape[0], :] = feats
            padded_masks[:masks.shape[0]] = masks
            padded_packed_sequences.append(padded_feats)
            padded_packed_masks.append(padded_masks)
        
        padded_packed_sequences = torch.stack(padded_packed_sequences, dim=0)
        padded_packed_masks = torch.stack(padded_packed_masks)
        return padded_packed_sequences, padded_packed_masks

    def reform_packed_sequences(self, hidden_states, masks):
        new_hidden_states = []
        max_length = 0
        if hidden_states.shape[1] != masks.shape[1]:
            hidden_states = hidden_states[:,hidden_states.shape[1]-masks.shape[1]:,:]
        for feats, mask in zip(hidden_states, masks):
            feats = feats[mask]
            new_hidden_states.append(feats)
            max_length = max(max_length, feats.shape[0])

        ref_hidden_states = []
        ref_masks = []
        for feats in new_hidden_states:
            padded_feats = torch.zeros(
                (max_length, feats.shape[1]), 
                dtype=feats.dtype,
                device=feats.device
            )
            padded_masks = torch.zeros(
                (max_length),
                dtype=torch.bool,
                device=feats.device
            )
            padded_feats[:feats.shape[0], :] = feats
            padded_masks[:feats.shape[0]] = True
            ref_hidden_states.append(padded_feats)
            ref_masks.append(padded_masks)
        
        ref_hidden_states = torch.stack(ref_hidden_states, dim=0)
        ref_masks = torch.stack(ref_masks)
        return ref_hidden_states, ref_masks

    def packing_encode_sequences(self, phrase_seqs, unit_seqs, ref_seqs):
        encode_feats = []
        encode_masks = []
        batch_size = len(phrase_seqs)
        for batch_idx in range(batch_size):
            p_seqs_batch = phrase_seqs[batch_idx]
            u_seqs_batch = unit_seqs[batch_idx]
            r_seqs_batch = ref_seqs[batch_idx]
            assert len(p_seqs_batch) == len(u_seqs_batch)
            assert len(p_seqs_batch) == len(r_seqs_batch)
            batch_feats = []
            batch_masks = []
            for p_feats, u_feats, r_feats in zip(p_seqs_batch, 
                                              u_seqs_batch, r_seqs_batch):
                batch_feats.append(p_feats)
                batch_feats.append(u_feats)
                batch_feats.append(r_feats)

                r_masks = torch.zeros(
                    (p_feats.shape[0] + u_feats.shape[0] + r_feats.shape[0]), 
                    dtype=torch.bool, 
                    device=r_feats.device
                )
                r_masks[-r_feats.shape[0]:] = True
                batch_masks.append(r_masks)

            batch_feats = torch.cat(batch_feats, dim=0)
            batch_masks = torch.cat(batch_masks, dim=0)
            encode_feats.append(batch_feats)
            encode_masks.append(batch_masks)

        max_len = max([feats.shape[0] for feats in encode_feats])
        padded_encode_feats = []
        padded_encode_masks = []
        for feats, masks in zip(encode_feats, encode_masks):
            padded_feats = torch.zeros(
                (max_len, feats.shape[1]),
                device=feats.device,
                dtype=feats.dtype
            )
            padded_masks = torch.zeros(
                (max_len),
                dtype=masks.dtype, 
                device=masks.device
            )
            padded_feats[:feats.shape[0], :] = feats
            padded_masks[:masks.shape[0]] = masks
            padded_encode_feats.append(padded_feats)
            padded_encode_masks.append(padded_masks)

        padded_encode_feats = torch.stack(padded_encode_feats, dim=0)
        padded_encode_masks = torch.stack(padded_encode_masks, dim=0)
        return padded_encode_feats, padded_encode_masks

    def forward_encode(
        self, 
        phrase_sequences, 
        unit_sequences, 
        ref_sequences,
        metas, 
        mode='loss'
    ):
        encode_feats, ref_masks = self.packing_encode_sequences(
            phrase_sequences, 
            unit_sequences, 
            ref_sequences
        )

        if 'visual' in self.config.mode:
            visual_hidden_states = metas['visual_hidden_states']
            if self.gradient_checkpointing and self.training:
                outputs = checkpoint(
                    self.model, 
                    encode_feats,
                    visual_hidden_states,
                    None,
                    mode,
                    use_reentrant=True
                )
            else:
                outputs = self.model(
                    encode_feats,
                    visual_hidden_states=visual_hidden_states,
                    mode=mode
                )            
        else:
            if self.gradient_checkpointing and self.training:
                outputs = checkpoint(
                    self.model, 
                    encode_feats,
                    None,
                    mode,
                    use_reentrant=True
                )
            else:
                outputs = self.model(
                    encode_feats,
                    mode=mode
                )
            
        last_hidden_states = outputs['last_hidden_states']
        ref_feats, ref_mask = self.reform_packed_sequences(last_hidden_states, ref_masks)
        return dict(
            ref_hidden_states = ref_feats,
            ref_mask = ref_mask
        )



    def forward_decode(
        self, 
        phrase_sequences, 
        unit_sequences, 
        ref_sequences,
        metas, 
        mode='loss'
    ):
        if self.config.packing:
            phrase_feats, _ = self.packing_sequences(phrase_sequences)
            unit_feats, _ = self.packing_sequences(unit_sequences)
            text_feats = torch.cat([phrase_feats, unit_feats], dim=1)
            ref_feats, ref_masks = self.packing_sequences(ref_sequences)
        else:
            phrase_feats = self.padding_sequences(phrase_sequences, self.config.phrase_max_length)
            unit_feats = self.padding_sequences(unit_sequences, self.config.unit_max_length)
            text_feats = torch.cat([phrase_feats, unit_feats], dim=1)
            ref_feats = self.padding_sequences(ref_sequences, self.config.ref_max_length)

        if self.gradient_checkpointing and self.training:
            outputs = checkpoint(
                self.model, 
                ref_feats,
                text_feats,
                None,
                mode,
                use_reentrant=True
            )
        else:
            outputs = self.model(
                ref_feats,
                text_feats,
                mode=mode
            )
        last_hidden_states = outputs['last_hidden_states']
        
        # re-organize hidden_states
        if self.config.packing:
            hidden_states, hidden_states_mask = self.reform_packed_sequences(last_hidden_states, ref_masks)
        else:
            hidden_states, hidden_states_mask = self.reform_padded_sequences(ref_sequences, last_hidden_states)
        return dict(
            hidden_states = outputs['hidden_states'],
            ref_hidden_states = hidden_states,
            ref_mask = hidden_states_mask
        )
    
    def forward_projector(
        self, 
        phrase_sequences, 
        unit_sequences, 
        ref_sequences,
        metas, 
        mode='loss'
    ):
        
        def _padding_sequences(ref_seqs):
            encode_feats = []
            encode_masks = []
            batch_size = len(ref_seqs)
            for batch_idx in range(batch_size):
                r_seqs_batch = ref_seqs[batch_idx]
                batch_feats = []
                batch_masks = []
                for r_feats in r_seqs_batch:
                    batch_feats.append(r_feats)

                    r_masks = torch.zeros(
                        (r_feats.shape[0]), 
                        dtype=torch.bool, 
                        device=r_feats.device
                    )
                    r_masks[-r_feats.shape[0]:] = True
                    batch_masks.append(r_masks)

                batch_feats = torch.cat(batch_feats, dim=0)
                batch_masks = torch.cat(batch_masks, dim=0)
                encode_feats.append(batch_feats)
                encode_masks.append(batch_masks)

            max_len = max([feats.shape[0] for feats in encode_feats])
            padded_encode_feats = []
            padded_encode_masks = []
            for feats, masks in zip(encode_feats, encode_masks):
                padded_feats = torch.zeros(
                    (max_len, feats.shape[1]),
                    device=feats.device,
                    dtype=feats.dtype
                )
                padded_masks = torch.zeros(
                    (max_len),
                    dtype=masks.dtype, 
                    device=masks.device
                )
                padded_feats[:feats.shape[0], :] = feats
                padded_masks[:masks.shape[0]] = masks
                padded_encode_feats.append(padded_feats)
                padded_encode_masks.append(padded_masks)

            padded_encode_feats = torch.stack(padded_encode_feats, dim=0)
            padded_encode_masks = torch.stack(padded_encode_masks, dim=0)
            return padded_encode_feats, padded_encode_masks


        # encode_feats, ref_masks  = _padding_sequences(ref_sequences)

        # outputs = self.model(encode_feats)
        # re-organize hidden_states

        # ref_feats, ref_mask = self.reform_packed_sequences(outputs, ref_masks)

        # return dict(
        #     hidden_states = outputs,
        #     ref_hidden_states = outputs,
        #     ref_mask = ref_masks
        # )
        outputs = self.model(ref_sequences)


        return dict(
            hidden_states = outputs,
            ref_hidden_states = outputs,
            ref_mask = None
        )


    def forward(
        self, 
        phrase_sequences, 
        unit_sequences, 
        ref_sequences,
        metas, 
        mode='loss'
    ):
        if 'encode' in self.config.mode:
            results = self.forward_encode(
                phrase_sequences, 
                unit_sequences, 
                ref_sequences,
                metas, 
                mode
            )
        elif self.config.mode == 'decode':
            results = self.forward_decode(
                phrase_sequences, 
                unit_sequences, 
                ref_sequences,
                metas, 
                mode
            )
        else:
            assert self.config.mode == 'projector'
            results = self.forward_projector(
                phrase_sequences, 
                unit_sequences, 
                ref_sequences,
                metas, 
                mode
            )
            
        return results