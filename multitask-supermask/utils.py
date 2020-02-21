import time
import pathlib

import torch
import torch.nn as nn
import models
import models.module_util as module_util
import torch.backends.cudnn as cudnn

from args import args

def set_gpu(model):
    if args.multigpu is None:
        args.device = torch.device("cpu")
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print(f"=> Parallelizing on {args.multigpu} gpus")
        torch.cuda.set_device(args.multigpu[0])
        args.gpu = args.multigpu[0]
        model = torch.nn.DataParallel(model, device_ids=args.multigpu).cuda(
            args.multigpu[0]
        )
        args.device = torch.cuda.current_device()
        cudnn.benchmark = True

    return model

def get_model():
    model = models.__dict__[args.model]()
    return model

def write_result_to_csv(**kwargs):
    results = pathlib.Path(args.log_dir) / "results.csv"

    if not results.exists():
        results.write_text(
            "Date Finished,"
            "Name,"
            "Current Val,"
            "Best Val\n"
        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (
                "{now}, "
                "{name}, "
                "{curr_acc1:.04f}, "
                "{best_acc1:.04f}\n"
            ).format(now=now, **kwargs)
        )

def write_hamming(writer, model, epoch):
    for n, m in model.named_modules():
        if not isinstance(m, nn.Conv2d): continue
        masks = {}
        for k, v in m.scores.items():
            masks[k] = module_util.GetSubnet.apply(v.abs(), args.sparsity)

        done = []
        for k1 in masks:
            done.append(k1)
            for k2 in masks:
                if k2 in done: continue
                hamming_distance = (masks[k1] != masks[k2]).float().mean().item()
                writer.add_scalar('hd/{}-{}-{}'.format(n, k1, k2), hamming_distance, epoch)

def log_hamming(model):
    results = pathlib.Path(args.log_dir) / "hamming.csv"
    if not results.exists():
        results.write_text(
            "Date Finished,"
            "Name,"
            "Distance\n"
        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")
    with open(results, "a+") as f:
        for n, m in model.named_modules():
            if not isinstance(m, nn.Conv2d): continue
            masks = {}
            for k, v in m.scores.items():
                masks[k] = module_util.GetSubnet.apply(v.abs(), args.sparsity)

            done = []
            for k1 in masks:
                done.append(k1)
                for k2 in masks:
                    if k2 in done: continue
                    hamming_distance = (masks[k1] != masks[k2]).float().mean().item()
                    name = args.name + '~layer={}~t1={}~t2={}'.format(n, k1, k2)
                    f.write(
                        (
                            "{now}, "
                            "{name}, "
                            "{distance:.04f}\n"
                        ).format(now=now, name=name, distance=hamming_distance)
                    )