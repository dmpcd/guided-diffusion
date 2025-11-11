#!/usr/bin/env python3
"""
Simple single-GPU ICG demo (no MPI required)
Demonstrates Independent Condition Guidance without needing a classifier
"""
import argparse
import os
import sys

import numpy as np
import torch as th

# Add the project to path
sys.path.insert(0, '/home/senum/projects/guided-diffusion/guided-diffusion')

from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()
    
    print("🚀 Starting ICG image generation...")
    print(f"   Device: {'CUDA' if th.cuda.is_available() else 'CPU'}")
    
    # Set device
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    
    # Set random seed for reproducibility BEFORE any random operations
    # This ensures class labels match other methods
    if args.seed is not None:
        th.manual_seed(args.seed)
        np.random.seed(args.seed)
        if th.cuda.is_available():
            th.cuda.manual_seed_all(args.seed)
    
    print("📦 Loading model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    # Load weights
    print(f"   Loading weights from: {args.model_path}")
    state_dict = th.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    
    print(f"✓ Model loaded successfully!")
    
    print(f"\n🎨 Generating {args.num_samples} samples with ICG...")
    print(f"   Resolution: {args.image_size}×{args.image_size}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   ICG Scale: {args.icg_scale}")
    print(f"   Steps: {args.timestep_respacing if args.timestep_respacing else args.diffusion_steps}")
    if args.seed is not None:
        print(f"   Seed: {args.seed}")
    print()
    
    all_images = []
    all_labels = []
    
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    for batch_idx in range(num_batches):
        print(f"📊 Batch {batch_idx + 1}/{num_batches}...")
        
        # Generate classes for this batch (matches simple_demo.py behavior)
        classes = th.randint(
            low=0, high=1000, size=(args.batch_size,), device=device
        )
        model_kwargs = {"y": classes}
        
        # Generate with ICG
        with th.no_grad():
            sample = diffusion.p_sample_loop_icg(
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                device=device,
                progress=True,
                icg_scale=args.icg_scale,
            )
        
        # Convert to uint8
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)  # BCHW -> BHWC
        sample = sample.contiguous().cpu().numpy()
        
        all_images.append(sample)
        all_labels.append(classes.cpu().numpy())
        
        print(f"   ✓ Generated batch {batch_idx + 1}")
    
    # Concatenate all batches
    arr = np.concatenate(all_images, axis=0)
    arr = arr[:args.num_samples]  # Trim to exact number requested
    
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[:args.num_samples]
    
    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_base = args.output_dir
    else:
        output_base = "."
    
    # Save
    shape_str = "x".join([str(x) for x in arr.shape])
    out_path = os.path.join(output_base, f"samples_{shape_str}.npz")
    
    print(f"\n💾 Saving to {out_path}...")
    np.savez(out_path, arr, label_arr)
    
    print(f"✅ Done! Generated {len(arr)} images with ICG")
    print(f"📁 Saved to: {out_path}")
    print(f"\n💡 ICG eliminated the need for a classifier while maintaining quality!")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=4,
        batch_size=4,
        use_ddim=False,
        model_path="models/64x64_diffusion.pt",
        icg_scale=1.5,
        output_dir="",
        seed=55,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
