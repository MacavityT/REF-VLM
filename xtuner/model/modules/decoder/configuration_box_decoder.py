# Copyright (c) OpenMMLab. All rights reserved.
from transformers import PretrainedConfig


class BoxDecoderConfig(PretrainedConfig):
    model_type = 'box_decoder'
    _auto_class = 'AutoConfig'

    def __init__(
        self,
        num_queries=100,
        decoder_layers=6,
        decoder_ffn_dim=2048,
        decoder_attention_heads=8,
        decoder_layerdrop=0.0,
        activation_function="relu",
        d_model=256,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        init_xavier_std=1.0,
        position_embedding_type="sine",
        bbox_loss_coefficient=5,
        giou_loss_coefficient=2,
        **kwargs,
    ):
        self.num_queries = num_queries
        self.d_model = d_model
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.init_xavier_std = init_xavier_std
        self.decoder_layerdrop = decoder_layerdrop
        self.position_embedding_type = position_embedding_type
        # Loss coefficients
        self.bbox_loss_coefficient = bbox_loss_coefficient
        self.giou_loss_coefficient = giou_loss_coefficient
        super().__init__(**kwargs)