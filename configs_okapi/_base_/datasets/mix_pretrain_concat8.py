_base_ = ['train_all_dataset.py']

data_args = dict(
    #
    train=dict(
        type='ConcatDataset',
        cfgs=[
            dict(
                type='SubSet',
                portion=1/20,
                do_shuffle=True,
                seed=42,
                cfg={{_base_.train_all_dataset.gc}},
            ),
            dict(
                type='SubSet',
                portion=1/20,
                do_shuffle=True,
                seed=43,
                cfg={{_base_.train_all_dataset.recvg}},
            ),

            {{_base_.train_all_dataset.llavacc3m}},
            {{_base_.train_all_dataset.llavalcs}},

            {{_base_.train_all_dataset.vqav2_train}},
            {{_base_.train_all_dataset.vqae_train}},
            {{_base_.train_all_dataset.vqax_train}},

            {{_base_.train_all_dataset.caption}},

            {{_base_.train_all_dataset.rec}},
            {{_base_.train_all_dataset.reg}},

            {{_base_.train_all_dataset.flickr}},

            {{_base_.train_all_dataset.vcr_q_ra}},
            {{_base_.train_all_dataset.vcr_qc_rac}},
            {{_base_.train_all_dataset.vcr_qac_r}},

            {{_base_.train_all_dataset.point_local_b}},
            {{_base_.train_all_dataset.point_local_p}},
            {{_base_.train_all_dataset.point_twice_oq_bp}},
            {{_base_.train_all_dataset.point_twice_sq_bp}},
            {{_base_.train_all_dataset.point_twice_gq_bp}},
            {{_base_.train_all_dataset.point_v7w_p}},
            {{_base_.train_all_dataset.point_v7w_b}},

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
