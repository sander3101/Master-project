import os
import time
import warnings
from argparse import ArgumentParser
from pathlib import Path

import datasets as ds
import torch
from accelerate import Accelerator
from rich import print
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import random as rnd


import utils.inference

warnings.filterwarnings("ignore", category=UserWarning)


def quarter_loss(img):
    image = img[0, :, :]
    N, M = image.shape
    mask = np.zeros((N, M))

    for i in range(N):
        if i % 4 != 0:
            mask[i, :] = 1

    return mask


def main(args):
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    diffusion, helper, cfg = utils.inference.setup_model(args.checkpoint)

    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        dynamo_backend=cfg.dynamo_backend,
        split_batches=True,
        even_batches=False,
        step_scheduler_with_optimizer=True,
    )
    device = accelerator.device

    if accelerator.is_local_main_process:
        print(f"{cfg=}")

    if args.batch_size == 1:
        num_workers = 1
    else:
        num_workers = cfg.num_workers

    dataset = ds.load_dataset(
        path="data/kitti_360",
        name=cfg.lidar_projection,
        split=ds.Split.TEST,
        num_proc=num_workers,
    ).with_format("torch")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
    )

    diffusion.to(device)
    sample_fn = torch.compile(
        diffusion.conditional_sample
    )  #### Use this for masked data
    helper, dataloader = accelerator.prepare(helper, dataloader)

    counter = 0
    times = []

    for batch in tqdm(
        dataloader,
        desc="sampling...",
        dynamic_ncols=True,
        disable=not accelerator.is_main_process,
    ):
        indices = batch["sample_id"].long().to(device)
        depth = batch["depth"].float().to(device)
        depth = helper.convert_depth(depth)
        depth = helper.normalize(depth)
        rflct = batch["reflectance"].float().to(device)
        rflct = helper.normalize(rflct)
        mask = batch["mask"].float().to(device)
        targets = torch.cat([depth, rflct], dim=1)

        ### 1/4 loss ###
        mask = (
            torch.stack([torch.tensor(quarter_loss(element)) for element in targets])
            .unsqueeze(1)
            .to(device)
        )

        if counter == 0:
            with torch.cuda.amp.autocast(enabled=True):
                results = sample_fn(
                    batch_size=mask.shape[0],
                    num_steps=args.num_steps,
                    progress=accelerator.is_main_process,
                    rng=torch.Generator(device=device).manual_seed(0),
                    mode="ddpm",
                    mask=mask.float(),
                    x_0=targets,
                ).clamp(-1, 1)

        elif counter > 0 and counter < 10:
            t0 = time.time()

            with torch.cuda.amp.autocast(enabled=True):
                results = sample_fn(
                    batch_size=mask.shape[0],
                    num_steps=args.num_steps,
                    progress=accelerator.is_main_process,
                    rng=torch.Generator(device=device).manual_seed(0),
                    mode="ddpm",
                    mask=mask.float(),
                    x_0=targets,
                ).clamp(-1, 1)

            t1 = time.time()
            times.append(t1 - t0)

        else:
            break

        counter += 1

    print("All times: ", times)
    print("Average time: ", sum(times) / len(times))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num_steps", type=int, default=8)
    # parser.add_argument("--batch_size", type=int, default=32 * 4)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    main(args)
