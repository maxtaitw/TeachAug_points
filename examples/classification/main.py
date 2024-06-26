import __init__
import os, argparse, yaml, numpy as np
from torch import multiprocessing as mp
from examples.classification.train import main as train
from examples.classification.pretrain import main as pretrain
from openpoints.utils import EasyConfig, dist_utils, find_free_port, generate_exp_directory, resume_exp_directory, Wandb
from examples.classification.train_autoaug import main as main_adaptpoint
from examples.classification.train_scanobjectnnc import main as main_scanobjectnnc
from examples.classification.train_modelnetc import main as main_modelnetc
from examples.classification.train_autoaug_modelnet import main as main_adaptpoint_modelnet
from examples.classification.train_teachaug import main as main_teachaugpoint
from examples.classification.train_teachaug_gen import main as main_teachaugpoint_gen
from examples.classification.train_teachaug_pretrained import main as main_teachaugpoint_pretrained
from examples.classification.train_teachaug_distill import main as main_teachaugpoint_distill
from examples.classification.train_teachaug_weight import main as main_teachaugpoint_weight
import datetime


if __name__ == "__main__":
    parser = argparse.ArgumentParser('S3DIS scene segmentation training')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)
    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)

    # init distributed env first, since logger depends on the dist info.
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1

    # init log dir
    cfg.task_name = args.cfg.split('.')[-2].split('/')[-2]
    cfg.exp_name = args.cfg.split('.')[-2].split('/')[-1]
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d%H%M")
    tags = [
        # cfg.task_name,  # task name (the folder of name under ./cfgs
        cfg.mode,
        cfg.exp_name,  # cfg file name
        # formatted_datetime,
        # f'ngpus{cfg.world_size}',
        # f'seed{cfg.seed}',
    ]
    opt_list = [] # for checking experiment configs from logging file
    for i, opt in enumerate(opts):
        if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'pretrain' not in opt and 'path' not in opt and 'wandb' not in opt and '/' not in opt:
            opt_list.append(opt)
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)
    cfg.opts = '-'.join(opt_list)

    if cfg.mode in ['resume', 'val', 'test']:
        resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path)
        cfg.wandb.tags = [cfg.mode]
    else:  # resume from the existing ckpt and reuse the folder.
        generate_exp_directory(cfg, tags, additional_id=os.environ.get('MASTER_PORT', None))
        cfg.wandb.tags = tags
    os.environ["JOB_LOG_DIR"] = cfg.log_dir
    cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, indent=2)
        os.system('cp %s %s' % (args.cfg, cfg.run_dir))
    cfg.cfg_path = cfg_path
    cfg.wandb.name = cfg.run_name

    if cfg.mode == 'pretrain':
        main = pretrain
    elif cfg.mode == 'adaptpoint':
        main = main_adaptpoint
    elif cfg.mode == 'scanobjectnnc':
        main = main_scanobjectnnc
    elif cfg.mode == 'modelnetc':
        main = main_modelnetc
    elif cfg.mode == 'adaptpoint_modelnet':
        main = main_adaptpoint_modelnet
    elif cfg.mode == 'teachaugpoint':
        main = main_teachaugpoint
    elif cfg.mode == 'teachaugpoint_gen':
        main = main_teachaugpoint_gen
    elif cfg.mode == 'teachaugpoint_pretrained':
        main = main_teachaugpoint_pretrained
    elif cfg.mode == 'teachaugpoint_distill':
        main = main_teachaugpoint_distill
    elif cfg.mode == 'teachaugpoint_weight':
        main = main_teachaugpoint_weight
    else:
        main = train



    # multi processing.
    if cfg.mp:
        port = find_free_port()
        cfg.dist_url = f"tcp://localhost:{port}"
        print('using mp spawn for distributed training')
        mp.spawn(main, nprocs=cfg.world_size, args=(cfg,))
    else:
        main(0, cfg, profile=args.profile)
