# Copyright (c) OpenMMLab. All rights reserved.
from transformers import PretrainedConfig
from .configuration_box_decoder import BoxDecoderConfig

class KeypointDecoderConfig(PretrainedConfig):

    def __init__(
        self,
        num_queries=30,
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
        num_body_points=17,
        keypoint_loss_coefficient=2,
        oks_loss_coefficient=2,
        cls_loss_coefficient=1,
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
        self.num_body_points = num_body_points
        # Loss coefficients
        self.keypoint_loss_coefficient = keypoint_loss_coefficient
        self.oks_loss_coefficient = oks_loss_coefficient
        self.cls_loss_coefficient = cls_loss_coefficient

class PoseDecoderConfig(PretrainedConfig):
    model_type = 'pose_decoder'

    def __init__(
        self,
        num_queries=20,
        encoder_input_transform='resize_concat',
        encoder_input_index=(0, 1, 2, 3),
        encoder_input_dim=1024,
        box_config=None,
        keypoint_config=None,
        use_group_matcher=True,
        use_auxiliary_loss=True,
        aux_loss_coefficient=0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_queries = num_queries
        self.encoder_input_transform = encoder_input_transform
        self.encoder_input_index = encoder_input_index
        self.encoder_input_dim = encoder_input_dim
        self.use_group_matcher = use_group_matcher
        self.use_auxiliary_loss = use_auxiliary_loss
        self.aux_loss_coefficient = aux_loss_coefficient
        self.box_config = BoxDecoderConfig(
            num_queries=num_queries,
            encoder_input_transform=encoder_input_transform,
            encoder_input_index=encoder_input_index,
            encoder_input_dim=encoder_input_dim,
            use_group_matcher=use_group_matcher,
            **box_config
            )
        self.keypoint_config = KeypointDecoderConfig(
            num_queries=num_queries,
            encoder_input_transform=encoder_input_transform,
            encoder_input_index=encoder_input_index,
            encoder_input_dim=encoder_input_dim,
            **keypoint_config
        )