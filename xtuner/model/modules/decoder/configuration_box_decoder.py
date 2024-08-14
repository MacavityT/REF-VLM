# Copyright (c) OpenMMLab. All rights reserved.
from transformers import PretrainedConfig


class BoxDecoderConfig(PretrainedConfig):
    model_type = 'box_decoder'
    _auto_class = 'AutoConfig'

    def __init__(
        self,
        num_queries=100,
        quries_input_dim=256,
        encoder_input_transform='resize_concat',
        encoder_input_index=(0, 1, 2, 3),
        encoder_input_dim=1024,
        decoder_layers=6,
        decoder_ffn_dim=2048,
        decoder_attention_heads=8,
        decoder_layerdrop=0.0,
        activation_function="relu",
        d_model=256,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        bbox_loss_coefficient=5,
        giou_loss_coefficient=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_queries = num_queries
        self.quries_input_dim = quries_input_dim
        self.encoder_input_transform = encoder_input_transform
        self.encoder_input_index = encoder_input_index
        self.encoder_input_dim = encoder_input_dim
        self.d_model = d_model
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.decoder_layerdrop = decoder_layerdrop
        # Loss coefficients
        self.bbox_loss_coefficient = bbox_loss_coefficient
        self.giou_loss_coefficient = giou_loss_coefficient
        