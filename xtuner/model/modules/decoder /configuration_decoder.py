# Copyright (c) OpenMMLab. All rights reserved.
from transformers import PretrainedConfig


class DecoderConfig(PretrainedConfig):
    model_type = 'decoder'
    _auto_class = 'AutoConfig'

    def __init__(
        self,
        visual_hidden_size=4096,
        llm_hidden_size=4096,
        depth=2,
        hidden_act='gelu',
        bias=True,
        **kwargs,
    ):
        self.visual_hidden_size = visual_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.depth = depth
        self.hidden_act = hidden_act
        self.bias = bias
        super().__init__(**kwargs)



class ProjectorConfig(PretrainedConfig):
    model_type = 'decoder'
    _auto_class = 'AutoConfig'

    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=14336,
        num_local_experts=8,
        num_experts_per_tok=2,
        hidden_act="silu",
        router_jitter_noise=0.0,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.hidden_act = hidden_act
        self.router_jitter_noise = router_jitter_noise
        super().__init__(**kwargs)

