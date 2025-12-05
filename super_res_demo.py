#!/usr/bin/env python3
"""
Simple single-GPU super-resolution demo (no MPI required)
Upsamples low-resolution images to high-resolution using diffusion models

Supports:
- 64×64 → 256×256 upsampling
- 128×128 → 512×512 upsampling
- Class-conditional generation
"""

import argparse
import os
import sys

# Add the project to path
sys.path.insert(0, '/home/senum/projects/guided-diffusion/guided-diffusion')

import numpy as np
import torch as th
import torch.nn.functional as F
from PIL import Image

from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()
    
    # Validate inputs
    if not args.base_samples:
        raise ValueError("--base_samples is required (path to low-res .npz file)")
    
    if not os.path.exists(args.base_samples):
        raise ValueError(f"Base samples file not found: {args.base_samples}")
    
    # Set random seed for reproducibility
    if args.seed is not None:
        th.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"🎲 Random seed set to: {args.seed}")
    
    print("🚀 Starting super-resolution...")
    print(f"   Device: {'CUDA' if th.cuda.is_available() else 'CPU'}")
    print(f"   Upsampling: {args.small_size}×{args.small_size} → {args.large_size}×{args.large_size}")
    
    # Set device
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    
    # Load base samples
    print(f"\n📂 Loading base samples from: {args.base_samples}")
    base_data = load_base_samples(args.base_samples, args.class_cond)
    print(f"   ✓ Loaded {len(base_data['images'])} low-resolution images")
    
    # Override classes if specified
    if args.class_idx and args.class_cond:
        custom_classes = parse_class_idx(args.class_idx, len(base_data['images']))
        base_data['labels'] = custom_classes
        print(f"   ✓ Using custom classes: {custom_classes.tolist()}")
    
    # Create model and diffusion
    print("\n📦 Loading super-resolution model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
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
    
    # Determine number of samples to generate
    num_to_generate = min(args.num_samples, len(base_data['images']))
    num_batches = (num_to_generate + args.batch_size - 1) // args.batch_size
    
    print(f"\n🎨 Upsampling {num_to_generate} images...")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Steps: {args.timestep_respacing if args.timestep_respacing else args.diffusion_steps}")
    print()
    
    all_images = []
    all_labels = []
    sample_idx = 0
    
    for batch_idx in range(num_batches):
        print(f"📊 Batch {batch_idx + 1}/{num_batches}...")
        
        # Get batch of low-res images
        batch_end = min(sample_idx + args.batch_size, num_to_generate)
        batch_low_res = base_data['images'][sample_idx:batch_end]
        
        # Prepare model kwargs
        model_kwargs = prepare_model_kwargs(
            batch_low_res,
            base_data.get('labels'),
            sample_idx,
            batch_end,
            args.class_cond,
            device
        )
        
        # Generate high-res samples
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        
        with th.no_grad():
            sample = sample_fn(
                model,
                (len(batch_low_res), 3, args.large_size, args.large_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                device=device,
                progress=True,
            )
        
        # Convert to uint8
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)  # BCHW -> BHWC
        sample = sample.contiguous().cpu().numpy()
        
        all_images.append(sample)
        if args.class_cond and 'labels' in base_data:
            batch_labels = base_data['labels'][sample_idx:batch_end]
            all_labels.append(batch_labels)
        
        sample_idx = batch_end
        print(f"   ✓ Upsampled batch {batch_idx + 1}")
    
    # Concatenate all batches
    arr = np.concatenate(all_images, axis=0)
    
    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_base = args.output_dir
    else:
        output_base = "."
    
    # Save as npz
    shape_str = "x".join([str(x) for x in arr.shape])
    out_path = os.path.join(output_base, f"upsampled_{shape_str}.npz")
    
    print(f"\n💾 Saving to {out_path}...")
    if args.class_cond and all_labels:
        label_arr = np.concatenate(all_labels, axis=0)
        np.savez(out_path, arr, label_arr)
    else:
        np.savez(out_path, arr)
    
    print(f"✅ Done! Upsampled {len(arr)} images")
    print(f"📁 Saved to: {out_path}")
    
    # Optionally save as PNG files
    if args.save_png:
        print(f"\n📸 Saving individual PNG files...")
        save_as_png(arr, output_base, args.large_size)


def parse_class_idx(class_idx_str, num_images):
    """
    Parse class indices from command line argument.
    
    Args:
        class_idx_str: Comma-separated class indices (e.g., "207,88,949,417")
        num_images: Number of images to generate classes for
    
    Returns:
        Numpy array of class indices
    """
    classes = [int(c.strip()) for c in class_idx_str.split(',')]
    
    # If fewer classes than images, repeat the pattern
    if len(classes) < num_images:
        classes = (classes * ((num_images // len(classes)) + 1))[:num_images]
    # If more classes than images, truncate
    elif len(classes) > num_images:
        classes = classes[:num_images]
    
    return np.array(classes, dtype=np.int64)


def load_base_samples(file_path, class_cond):
    """
    Load low-resolution samples from .npz file.
    
    Args:
        file_path: Path to .npz file containing base samples
        class_cond: Whether to expect class labels
    
    Returns:
        Dictionary with 'images' and optionally 'labels'
    """
    with open(file_path, 'rb') as f:
        data = np.load(f)
        images = data['arr_0']
        
        result = {'images': images}
        
        if class_cond and 'arr_1' in data.files:
            result['labels'] = data['arr_1']
        elif class_cond:
            print("   ⚠️  Warning: class_cond=True but no labels found in base samples")
        
        return result


def prepare_model_kwargs(batch_low_res, all_labels, start_idx, end_idx, class_cond, device):
    """
    Prepare model keyword arguments for super-resolution.
    
    Args:
        batch_low_res: Batch of low-resolution images (numpy array)
        all_labels: All class labels (numpy array or None)
        start_idx: Starting index in the full dataset
        end_idx: Ending index in the full dataset
        class_cond: Whether to include class conditioning
        device: PyTorch device
    
    Returns:
        Dictionary of model kwargs
    """
    # Convert low-res images to tensor
    low_res = th.from_numpy(batch_low_res).float()
    low_res = low_res / 127.5 - 1.0  # Normalize to [-1, 1]
    low_res = low_res.permute(0, 3, 1, 2)  # BHWC -> BCHW
    low_res = low_res.to(device)
    
    model_kwargs = {'low_res': low_res}
    
    # Add class labels if available
    if class_cond and all_labels is not None:
        batch_labels = all_labels[start_idx:end_idx]
        model_kwargs['y'] = th.from_numpy(batch_labels).to(device)
    
    return model_kwargs


def save_as_png(images, output_dir, resolution):
    """
    Save images as individual PNG files.
    
    Args:
        images: Numpy array of images (N, H, W, 3)
        output_dir: Directory to save images
        resolution: Image resolution for filename
    """
    png_dir = os.path.join(output_dir, f'upsampled_{resolution}x{resolution}')
    os.makedirs(png_dir, exist_ok=True)
    
    for i, img in enumerate(images):
        save_path = os.path.join(png_dir, f'upsampled_{i+1:04d}.png')
        Image.fromarray(img).save(save_path)
        print(f'   ✓ {save_path}')
    
    print(f"   Saved {len(images)} PNG files to {png_dir}/")


def create_argparser():
    """
    Create argument parser with defaults for super-resolution.
    """
    defaults = dict(
        clip_denoised=True,
        num_samples=10,
        batch_size=4,
        use_ddim=False,
        model_path="",
        base_samples="",
        output_dir="outputs/super_res",
        seed=42,
        save_png=False,  # Save PNG files in addition to .npz
        class_idx="",  # Comma-separated class indices (e.g., "207,88,949,417")
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser(
        description="Super-resolution demo using diffusion models"
    )
    add_dict_to_argparser(parser, defaults)
    
    return parser


if __name__ == "__main__":
    main()
