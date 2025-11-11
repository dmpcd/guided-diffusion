"""
Generate samples using Independent Condition Guidance (ICG).

ICG is a training-free method that eliminates the need for a classifier
by using the diffusion model itself to provide guidance. It improves sample
quality without requiring a separate classifier model.

Based on "No Training, No Problem: Rethinking Classifier-Free Guidance for 
Diffusion Models" (Sadat et al., ICLR 2025).
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log(f"sampling with ICG (scale={args.icg_scale})...")
    all_images = []
    all_labels = []
    
    while len(all_images) * args.batch_size < args.num_samples:
        # Generate random class labels for conditioning
        classes = th.randint(
            low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
        )
        model_kwargs = {"y": classes}
        
        # Use ICG sampling loop
        sample = diffusion.p_sample_loop_icg(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            progress=True,
            icg_scale=args.icg_scale,
        )
        
        # Convert to uint8 for saving
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        # Gather samples from all processes (for distributed training)
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        
        # Gather labels
        gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, classes)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    # Concatenate and trim to exact number of samples
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]
    
    # Save results (only on rank 0 for distributed training)
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        icg_scale=1.5,  # Default ICG guidance scale
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
