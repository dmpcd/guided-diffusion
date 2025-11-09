#!/usr/bin/env python3
"""
Simple single-GPU image generation demo (no MPI required)
Supports both unconditional and classifier-guided sampling
"""
import argparse
import os
import sys

# Add the project to path
sys.path.insert(0, '/home/senum/projects/guided-diffusion/guided-diffusion')

import numpy as np
import torch as th
import torch.nn.functional as F

from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)

def main():
    args = create_argparser().parse_args()
    
    # Set random seed for reproducibility
    if args.seed is not None:
        th.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"🎲 Random seed set to: {args.seed}")
    
    print("🚀 Starting image generation...")
    print(f"   Device: {'CUDA' if th.cuda.is_available() else 'CPU'}")
    
    # Set device
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    
    # Determine if using classifier guidance
    use_classifier = args.classifier_path and os.path.exists(args.classifier_path)
    
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
    
    # Load classifier if provided
    classifier = None
    if use_classifier:
        print("📦 Loading classifier for guidance...")
        print(f"   Loading classifier from: {args.classifier_path}")
        classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
        state_dict = th.load(args.classifier_path, map_location="cpu")
        classifier.load_state_dict(state_dict)
        classifier.to(device)
        if args.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()
        print(f"✓ Classifier loaded successfully!")
        print(f"   Guidance scale: {args.classifier_scale}")
    
    print(f"\n🎨 Generating {args.num_samples} samples...")
    print(f"   Resolution: {args.image_size}×{args.image_size}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Steps: {args.timestep_respacing if args.timestep_respacing else args.diffusion_steps}")
    print(f"   Mode: {'Classifier-guided' if use_classifier else 'Unconditional'}")
    if args.seed is not None:
        print(f"   Seed: {args.seed} (for reproducible class selection)")
    print()
    
    # Define classifier guidance function
    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale
    
    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)
    
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
            if use_classifier:
                # Use classifier guidance
                sample = sample_fn(
                    model_fn,
                    (args.batch_size, 3, args.image_size, args.image_size),
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    cond_fn=cond_fn,
                    device=device,
                    progress=True,
                )
            else:
                # Standard sampling without classifier
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
        classifier_path="",
        classifier_scale=1.0,
        output_dir="",
        seed=42,  # Default seed for reproducibility
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
