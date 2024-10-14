# Copyright (c) OpenMMLab. All rights reserved.
from transformers import PretrainedConfig


class MaskDecoderConfig(PretrainedConfig):
    model_type = 'mask_decoder'
    _auto_class = 'AutoConfig'

    def __init__(
        self,
        num_queries=30,
        quries_input_dim=4096,
        encoder_input_transform='multiple_select',
        encoder_input_index=[3, 2, 1], 
        encoder_input_dim=[1024, 1024, 1024],
        decoder_layers=10,
        decoder_ffn_dim=2048,
        decoder_attention_heads=8,
        pre_norm=False,
        activation_function="relu",
        d_model=256,
        dropout=0.0,
        encoder_layers=6, 
        feature_size=256,
        mask_feature_size=256,
        feature_strides=[4, 8, 16, 32],
        common_stride=4,
        encoder_feedforward_dim=1024,
        use_group_matcher=True,
        mask_loss_coefficient=20,
        dice_loss_coefficient=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_queries = num_queries
        self.quries_input_dim = quries_input_dim
        self.encoder_input_transform = encoder_input_transform
        self.encoder_input_index = encoder_input_index
        self.encoder_input_dim = encoder_input_dim
        self.decoder_layers = decoder_layers
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_attention_heads = decoder_attention_heads
        self.pre_norm = pre_norm
        self.activation_function = activation_function
        self.d_model = d_model
        self.dropout = dropout
        self.encoder_layers = encoder_layers
        self.feature_size = feature_size
        self.mask_feature_size = mask_feature_size
        self.feature_strides = feature_strides
        self.common_stride = common_stride
        self.encoder_feedforward_dim = encoder_feedforward_dim
        self.use_group_matcher = use_group_matcher
        self.mask_loss_coefficient = mask_loss_coefficient
        self.dice_loss_coefficient = dice_loss_coefficient
        