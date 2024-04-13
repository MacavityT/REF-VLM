_base_ = ['train_all_dataset.py']

data_args = dict(
    #
    train=dict(
        type='ConcatDataset',
        cfgs=[
            {{_base_.train_all_dataset.rec}},
            {{_base_.train_all_dataset.recvg}},
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
