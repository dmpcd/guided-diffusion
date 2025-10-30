#!/usr/bin/env python3
"""
Simple single-GPU image generation demo (no MPI required)
"""
import argparse
import os
import sys

# Add the project to path
sys.path.insert(0, '/home/senum/projects/guided-diffusion/guided-diffusion')

import numpy as np
import torch as th

from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def main():
    args = create_argparser().parse_args()
    
    print("🚀 Starting image generation...")
    print(f"   Device: {'CUDA' if th.cuda.is_available() else 'CPU'}")
    
    # Set device
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    
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
    print(f"\n🎨 Generating {args.num_samples} samples...")
    print(f"   Resolution: {args.image_size}×{args.image_size}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Steps: {args.timestep_respacing if args.timestep_respacing else args.diffusion_steps}")
    print()
    
    all_images = []
    all_labels = []
    
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    for batch_idx in range(num_batches):
        print(f"📊 Batch {batch_idx + 1}/{num_batches}...")
        
        model_kwargs = {}
        if args.class_cond:
            # Random classes
            classes = th.randint(
                low=0, high=1000, size=(args.batch_size,), device=device
            )
            model_kwargs["y"] = classes
        
        # Sample
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        
        with th.no_grad():
            sample = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                progress=True,
            )
        
        # Convert to uint8
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)  # BCHW -> BHWC
        sample = sample.contiguous().cpu().numpy()
        
        all_images.append(sample)
        if args.class_cond:
            all_labels.append(classes.cpu().numpy())
        
        print(f"   ✓ Generated batch {batch_idx + 1}")
    
    # Concatenate all batches
    arr = np.concatenate(all_images, axis=0)
    arr = arr[:args.num_samples]  # Trim to exact number requested
    
    # Save
    shape_str = "x".join([str(x) for x in arr.shape])
    out_path = f"samples_{shape_str}.npz"
    
    print(f"\n💾 Saving to {out_path}...")
    if args.class_cond and all_labels:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[:args.num_samples]
        np.savez(out_path, arr, label_arr)
    else:
        np.savez(out_path, arr)
    
    print(f"✅ Done! Generated {len(arr)} images")
    print(f"📁 Saved to: {out_path}")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=4,
        batch_size=4,
        use_ddim=False,
        model_path="models/64x64_diffusion.pt",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
