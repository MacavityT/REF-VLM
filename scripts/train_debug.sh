
python -m debugpy --wait-for-client --listen 127.0.0.1:5577 -m xtuner.tools.train configs/train_stage2_vd_adapter.py --deepspeed deepspeed_zero2 --work-dir work_dirs/debug