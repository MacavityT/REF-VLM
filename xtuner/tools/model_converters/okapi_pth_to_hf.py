# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import shutil

from mmengine.config import Config, DictAction
from mmengine.fileio import PetrelBackend, get_file_backend

from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint
from xtuner.registry import BUILDER
from xtuner.model import OkapiModel


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert the pth model to HuggingFace model')
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('pth_model', help='pth model file')
    parser.add_argument(
        'save_dir', help='the directory to save HuggingFace model')
    parser.add_argument(
        '--fp32',
        action='store_true',
        help='Save LLM in fp32. If not set, fp16 will be used by default.')
    parser.add_argument(
        '--max-shard-size',
        type=str,
        default='2GB',
        help='Only applicable for LLM. The maximum size for '
        'each sharded checkpoint.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # parse config
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f'Cannot find {args.config}')

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    model_name = cfg.model.type if isinstance(cfg.model.type,
                                              str) else cfg.model.type.__name__
    if 'LLaVAModel' in model_name:
        cfg.model.pretrained_pth = None

    model = BUILDER.build(cfg.model)

    backend = get_file_backend(args.pth_model)
    if isinstance(backend, PetrelBackend):
        from xtuner.utils.fileio import patch_fileio
        with patch_fileio():
            state_dict = guess_load_checkpoint(args.pth_model)
    else:
        state_dict = guess_load_checkpoint(args.pth_model)

    model.load_state_dict(state_dict, strict=False)
    model.llm.config.use_cache = True

    print(f'Load PTH model from {args.pth_model}')

    assert 'OkapiModel' in model_name
    if cfg.model.get('llm') and (not cfg.model.get('freeze_llm', False)
                                    or cfg.model.get('llm_lora')):
        if 'PeftModel' in model.llm.__class__.__name__:
            llm_path = osp.join(args.save_dir, 'llm_adapter')
            print(f'Saving LLM adapter to {llm_path}')
        else:
            llm_path = args.save_dir
            print(f'Saving LLM tokenizer to {llm_path}')
            if hasattr(model, 'vpt_encoder'):
                tokenizer = OkapiModel._prepare_tokenizer(cfg.tokenizer)
            else:
                tokenizer = BUILDER.build(cfg.tokenizer)
            tokenizer.save_pretrained(llm_path)
            print(f'Saving LLM to {llm_path}')
        if not args.fp32:
            print('Convert LLM to float16')
            model.llm.half()
        model.llm.save_pretrained(
            llm_path, max_shard_size=args.max_shard_size)

    if cfg.model.get('visual_encoder') and (
            not cfg.model.get('freeze_visual_encoder', False)
            or cfg.model.get('visual_encoder_lora')):
        if 'PeftModel' in model.visual_encoder.__class__.__name__:
            visual_encoder_path = osp.join(args.save_dir,
                                            'visual_encoder_adapter')
            print(
                f'Saving visual_encoder adapter to {visual_encoder_path}')
        else:
            visual_encoder_path = osp.join(args.save_dir, 'visual_encoder')
            print('Saving visual_encoder image_processor to'
                    f'{visual_encoder_path}')
            image_processor = BUILDER.build(cfg.image_processor)
            image_processor.save_pretrained(visual_encoder_path)
            print(f'Saving visual_encoder to {visual_encoder_path}')
        model.visual_encoder.save_pretrained(
            visual_encoder_path, max_shard_size=args.max_shard_size)

    if hasattr(model, 'projector'):
        projector_path = osp.join(args.save_dir, 'projector')
        print(f'Saving projector to {projector_path}')
        model.projector.save_pretrained(
            projector_path, max_shard_size=args.max_shard_size)
        
    if hasattr(model, 'vpt_encoder'):
        vpt_encoder_path = osp.join(args.save_dir, 'vpt_encoder')
        print(f'Saving vpt_encoder to {vpt_encoder_path}')
        model.vpt_encoder.save_pretrained(
            vpt_encoder_path, max_shard_size=args.max_shard_size)
        
    if hasattr(model, 'visual_sync_tuner'):
        if model.visual_sync_tuner is not None:
            visual_sync_tuner_path = osp.join(args.save_dir, 'visual_sync_tuner')
            print(f'Saving visual_sync_tuner to {visual_sync_tuner_path}')
            model.visual_sync_tuner.save_pretrained(
                visual_sync_tuner_path, max_shard_size=args.max_shard_size)
            
    if hasattr(model, 'ref_adapter'):
        if model.ref_adapter is not None:
            ref_adapter_path = osp.join(args.save_dir, 'ref_adapter')
            print(f'Saving ref_adapter to {ref_adapter_path}')
            model.ref_adapter.save_pretrained(
                ref_adapter_path, max_shard_size=args.max_shard_size)
            
    if hasattr(model, 'visual_decoder'):
        if model.visual_decoder is not None:
            for name, decoder in model.visual_decoder.items():
                visual_decoder_path = osp.join(args.save_dir, f'{name}_decoder')
                print(f'Saving {name}_decoder to {visual_decoder_path}')
                decoder.save_pretrained(visual_decoder_path, max_shard_size=args.max_shard_size)


    shutil.copyfile(args.config, osp.join(args.save_dir, 'xtuner_config.py'))
    print('All done!')


if __name__ == '__main__':
    main()
