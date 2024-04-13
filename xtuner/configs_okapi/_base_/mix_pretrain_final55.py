_base_ = ['train_all_dataset.py']

data_args = dict(
    #
    train=dict(
        type='InterleaveDateset',
        probabilities=[0.5, 0.5],
        seed=None,
        stopping_strategy='first_exhausted',
        cfgs=[
            dict(
                type='ConcatDatasetWithShuffle',
                cfgs=[
                    {{_base_.train_all_dataset.instruct}},
                    {{_base_.train_all_dataset.gpt4gen_qbc}},
                    {{_base_.train_all_dataset.gpt4gen_rd_qbc}},
                ]
            ),
            dict(
                type='InterleaveDateset',
                probabilities=[1 / 7] * 7,
                seed=None,
                stopping_strategy='first_exhausted',
                cfgs=[
                    {{_base_.train_all_dataset.flickr}},
                    {{_base_.train_all_dataset.rec}},
                    dict(
                        type='ConcatDatasetWithShuffle',
                        seed=43,
                        cfgs=[
                            {{_base_.train_all_dataset.vqav2_train}},
                            {{_base_.train_all_dataset.vqae_train}},
                            {{_base_.train_all_dataset.vqax_train}},
                        ],
                    ),
                    dict(
                        type='ConcatDatasetWithShuffle',
                        seed=44,
                        cfgs=[
                            {{_base_.train_all_dataset.vcr_q_ra}},
                            {{_base_.train_all_dataset.vcr_qc_rac}},
                            {{_base_.train_all_dataset.vcr_qac_r}},
                        ],
                    ),
                    dict(
                        type='ConcatDatasetWithShuffle',
                        seed=45,
                        portion=3,
                        cfgs=[
                            {{_base_.train_all_dataset.point_local_b}},
                            {{_base_.train_all_dataset.point_local_p}},
                        ]
                    ),
                    dict(
                        type='ConcatDatasetWithShuffle',
                        seed=46,
                        cfgs=[
                            {{_base_.train_all_dataset.point_twice_oq_bp}},
                            {{_base_.train_all_dataset.point_twice_sq_bp}},
                            {{_base_.train_all_dataset.point_twice_gq_bp}},
                            {{_base_.train_all_dataset.point_v7w_p}},
                            {{_base_.train_all_dataset.point_v7w_b}},
                        ]
                    ),
                    dict(
                        type='ConcatDatasetWithShuffle',
                        seed=47,
                        cfgs=[
                            {{_base_.train_all_dataset.reg}},
                            {{_base_.train_all_dataset.caption}},
                            dict(
                                type='SubSet',
                                portion=2 / 3,
                                do_shuffle=True,
                                seed=40,
                                cfg={{_base_.train_all_dataset.llavacc3m}},
                            ),
                            dict(
                                type='SubSet',
                                portion=2 / 3,
                                do_shuffle=True,
                                seed=41,
                                cfg={{_base_.train_all_dataset.llavalcs}},
                            ),
                            dict(
                                type='SubSet',
                                portion=1 / 15,
                                do_shuffle=True,
                                seed=42,
                                cfg={{_base_.train_all_dataset.gc}},
                            ),
                        ]
                    )
                ],
            )
        ],
    ),
    validation=None,
    test=None,

    # compute_metric
    compute_metric=None,

    # padding collator kwargs
    collator_kwargs=dict(
        padding=True,
        max_length=1024,
    ),

    # generate config
    gen_kwargs=dict(
        max_new_tokens=1024,
        num_beams=1,
    ),
)
