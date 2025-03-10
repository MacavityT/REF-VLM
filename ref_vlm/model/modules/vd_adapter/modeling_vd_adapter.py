# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import warnings
import math
from typing import List, Optional, Tuple, Union
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from torch.utils.checkpoint import checkpoint

from .configuration_vd_adapter import VDAdapterConfig
from .transformer import TwoWayTransformer, MLPBlock
from .prompt_encoder import PositionEmbeddingRandom

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

class VDAdapterModel(PreTrainedModel):

    _auto_class = 'AutoModel'
    config_class = VDAdapterConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True

    def __init__(self, config: VDAdapterConfig):
        super().__init__(config)
        self.config = config
        
        num_queries= config.ref_max_length + config.phrase_max_length
        d_input_image = config.d_input_image
        d_input_token = config.d_input_token
        d_model = config.d_model

        self.in_proj_image = MLPBlock(embedding_dim=d_input_image, mlp_dim=8192)
        self.in_proj_text = MLPBlock(embedding_dim=d_input_token, mlp_dim=8192)
        # self.in_proj_ref = MLPBlock(embedding_dim=d_input_token, mlp_dim=8192)
        
        self.pe_image = PositionEmbeddingRandom(d_model // 2)
        self.pe_queries = PositionalEncoding(
            d_model, 
            dropout=config.dropout, 
            max_len=num_queries,
            batch_first=True
        )
        # self.ref_embedding = nn.Embedding(config.ref_max_length, config.d_model)
        self.text_embedding = nn.Embedding(config.phrase_max_length + config.ref_max_length, config.d_model)
        self.model = TwoWayTransformer(
                depth=config.num_layers,
                embedding_dim=config.d_model,
                mlp_dim=config.d_ffn,
                num_heads=config.n_heads,
            )
        self.output_tokens = nn.Embedding(config.ref_max_length, config.d_model)
        self.post_init()

    def enable_input_require_grads(self):
        
        def make_inputs_require_grad(module, input, output):
            if isinstance(output, dict):
                for key in output.keys():
                    if isinstance(output[key], torch.Tensor):
                        output[key].requires_grad_(True)
            elif isinstance(output, List) or isinstance(output, Tuple):
                for item in output:
                    item.requires_grad_(True)
            else:
                output.requires_grad_(True)
        
        self.model.register_forward_hook(make_inputs_require_grad)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, VDAdapterModel):
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

    # def padding_sequences_2(self, sequences):
    #     padded_sequences = []
    #     for seqs in sequences:
    #         seqs = torch.cat(seqs, dim=0)
    #         padded_sequences.append(seqs)
    #     return padded_sequences
    
    # def reform_padded_sequences(self, ef_sequences, hidden_states):
    #     pass

    def blc2bchw(self, x):
        if len(x.shape) == 3:
            b, l, c = x.shape
            grid_size = int(math.sqrt(l))
            x = x.view(b, grid_size, grid_size, c)
            x = x.permute(0, 3, 1, 2) # b, c, h, w
        return x
    
    def bchw2blc(self, x):
        if len(x.shape) == 4:
            b, c, h, w = x.shape
            x = x.permute(0, 2, 3, 1).view(b, -1, c) # b, L, c
        return x

    def forward(
        self,
        image_embeddings,
        phrase_sequences, 
        ref_sequences,
        metas, 
        mode='loss'
    ):
        # output tokens
        text_feats = self.padding_sequences(phrase_sequences, self.config.phrase_max_length)
        ref_feats = self.padding_sequences(ref_sequences, self.config.ref_max_length)

        # ref_feats = self.in_proj_ref(ref_feats)
        # text_feats = self.in_proj_text(text_feats)
        
        # # tokens embedding
        # text_feats += self.text_embedding.weight
        # ref_feats += self.ref_embedding.weight
        # output_tokens = torch.cat([ref_feats, text_feats], dim=1)
        # output_tokens = self.pe_queries(output_tokens)

        output_tokens = torch.cat([ref_feats, text_feats], dim=1)
        output_tokens = self.in_proj_text(output_tokens)
        output_tokens += self.text_embedding.weight
        output_tokens = self.pe_queries(output_tokens)

        # image embedding
        image_embeddings = self.in_proj_image(image_embeddings)
        length = image_embeddings.shape[1]
        grid_size = int(math.sqrt(length))

        image_embeddings_expand = []
        for seqs, curr_embedding in zip(ref_sequences, image_embeddings):
            num_ref_group = len(seqs)
            batch_embedding = torch.repeat_interleave(curr_embedding.unsqueeze(0), num_ref_group, dim=0)
            image_embeddings_expand.append(batch_embedding)
        image_embeddings_expand = torch.cat(image_embeddings_expand, dim=0)
        image_embeddings_expand = self.blc2bchw(image_embeddings_expand)
        image_pos = self.pe_image((grid_size, grid_size)).unsqueeze(0)
        image_pos_expand = torch.repeat_interleave(image_pos, output_tokens.shape[0], dim=0)

        if self.gradient_checkpointing and self.training:
            output_tokens, image_embeddings_expand = checkpoint(
                self.model, 
                image_embeddings_expand,
                image_pos_expand,
                output_tokens,
                use_reentrant=True
            )
        else:
            output_tokens, image_embeddings_expand = self.model(
                image_embeddings_expand,
                image_pos_expand,
                output_tokens,
            )
        
        # fusion of image embeddings
        start = 0
        image_embeddings_fused = []
        for seqs in ref_sequences:
            num_ref_group = len(seqs)
            batch_embedding_fused = image_embeddings_expand[start:start+num_ref_group, ...]
            batch_embedding_fused = torch.mean(batch_embedding_fused, dim=0)
            image_embeddings_fused.append(batch_embedding_fused)
            start += num_ref_group
        image_embeddings_fused = torch.stack(image_embeddings_fused, dim=0)

        # re-organize hidden_states
        hidden_states, hidden_states_mask = self.reform_padded_sequences(
            ref_sequences, 
            output_tokens[:, :self.config.ref_max_length, :]
        )
        return dict(
            image_embeddings = image_embeddings_fused,
            ref_hidden_states = hidden_states,
            ref_mask = hidden_states_mask
        )
    
    # def forward(
    #     self,
    #     image_embeddings,
    #     phrase_sequences, 
    #     ref_sequences,
    #     metas, 
    #     mode='loss'
    # ):
    #     # output tokens
    #     text_feats = self.padding_sequences(phrase_sequences, self.config.phrase_max_length)
    #     ref_feats = self.padding_sequences(ref_sequences, self.config.ref_max_length)

    #     for 