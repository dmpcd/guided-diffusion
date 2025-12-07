#!/usr/bin/env python3
"""
Windows-compatible image generation script for Guided Diffusion
Run this instead of the .sh files on Windows

Usage:
    python generate_images_windows.py --resolution 64 --num_samples 4
    python generate_images_windows.py --resolution 128 --num_samples 4 --use_classifier
    python generate_images_windows.py --resolution 512 --num_samples 1 --use_classifier
"""
import argparse
import os
import sys
import subprocess

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# Add project to path
sys.path.insert(0, SCRIPT_DIR)

def check_models(resolution, use_classifier):
    """Check if required models exist"""
    model_dir = os.path.join(SCRIPT_DIR, "models")
    
    diffusion_model = os.path.join(model_dir, f"{resolution}x{resolution}_diffusion.pt")
    classifier_model = os.path.join(model_dir, f"{resolution}x{resolution}_classifier.pt")
    
    missing = []
    if not os.path.exists(diffusion_model):
        missing.append(f"{resolution}x{resolution}_diffusion.pt")
    if use_classifier and not os.path.exists(classifier_model):
        missing.append(f"{resolution}x{resolution}_classifier.pt")
    
    return missing

def get_model_config(resolution):
    """Get model configuration based on resolution"""
    configs = {
        64: {
            "attention_resolutions": "32,16,8",
            "class_cond": True,
            "diffusion_steps": 1000,
            "dropout": 0.1,
            "image_size": 64,
            "learn_sigma": True,
            "noise_schedule": "cosine",
            "num_channels": 192,
            "num_head_channels": 64,
            "num_res_blocks": 3,
            "resblock_updown": True,
            "use_new_attention_order": True,
            "use_scale_shift_norm": True,
            "classifier_depth": 4,
            "classifier_scale": 1.0,
        },
        128: {
            "attention_resolutions": "32,16,8",
            "class_cond": True,
            "diffusion_steps": 1000,
            "dropout": 0.0,
            "image_size": 128,
            "learn_sigma": True,
            "noise_schedule": "linear",
            "num_channels": 256,
            "num_heads": 4,
            "num_res_blocks": 2,
            "resblock_updown": True,
            "use_scale_shift_norm": True,
            "classifier_depth": 2,
            "classifier_width": 128,
            "classifier_attention_resolutions": "32,16,8",
            "classifier_scale": 0.5,
        },
        512: {
            "attention_resolutions": "32,16,8",
            "class_cond": True,
            "diffusion_steps": 1000,
            "dropout": 0.0,
            "image_size": 512,
            "learn_sigma": True,
            "noise_schedule": "linear",
            "num_channels": 256,
            "num_head_channels": 64,
            "num_res_blocks": 2,
            "resblock_updown": True,
            "use_scale_shift_norm": True,
            "classifier_depth": 2,
            "classifier_width": 128,
            "classifier_attention_resolutions": "32,16,8",
            "classifier_scale": 4.0,
        }
    }
    return configs.get(resolution, configs[64])

def generate_images(args):
    """Generate images using simple_demo.py"""
    import numpy as np
    import torch as th
    import torch.nn.functional as F
    from PIL import Image
    
    from guided_diffusion.script_util import (
        model_and_diffusion_defaults,
        classifier_defaults,
        create_model_and_diffusion,
        create_classifier,
    )
    
    config = get_model_config(args.resolution)
    
    # Set device
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    if device.type == 'cpu':
        print("⚠️  Warning: Running on CPU will be very slow!")
        print("   For faster generation, ensure CUDA is available.")
    
    # Set random seed
    if args.seed is not None:
        th.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"🎲 Random seed: {args.seed}")
    
    # Model paths
    model_path = os.path.join(SCRIPT_DIR, "models", f"{args.resolution}x{args.resolution}_diffusion.pt")
    classifier_path = os.path.join(SCRIPT_DIR, "models", f"{args.resolution}x{args.resolution}_classifier.pt") if args.use_classifier else None
    
    print(f"\n📦 Loading {args.resolution}x{args.resolution} diffusion model...")
    
    # Create model arguments
    model_args = argparse.Namespace(
        image_size=config["image_size"],
        num_channels=config["num_channels"],
        num_res_blocks=config["num_res_blocks"],
        num_heads=config.get("num_heads", -1),
        num_head_channels=config.get("num_head_channels", -1),
        num_heads_upsample=-1,
        attention_resolutions=config["attention_resolutions"],
        channel_mult="",
        dropout=config["dropout"],
        class_cond=config["class_cond"],
        use_checkpoint=False,
        use_scale_shift_norm=config["use_scale_shift_norm"],
        resblock_updown=config["resblock_updown"],
        use_fp16=args.use_fp16,
        use_new_attention_order=config.get("use_new_attention_order", False),
        learn_sigma=config["learn_sigma"],
        diffusion_steps=config["diffusion_steps"],
        noise_schedule=config["noise_schedule"],
        timestep_respacing=str(args.steps),
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )
    
    # Create model and diffusion
    model, diffusion = create_model_and_diffusion(
        image_size=model_args.image_size,
        class_cond=model_args.class_cond,
        learn_sigma=model_args.learn_sigma,
        num_channels=model_args.num_channels,
        num_res_blocks=model_args.num_res_blocks,
        channel_mult=model_args.channel_mult,
        num_heads=model_args.num_heads,
        num_head_channels=model_args.num_head_channels,
        num_heads_upsample=model_args.num_heads_upsample,
        attention_resolutions=model_args.attention_resolutions,
        dropout=model_args.dropout,
        diffusion_steps=model_args.diffusion_steps,
        noise_schedule=model_args.noise_schedule,
        timestep_respacing=model_args.timestep_respacing,
        use_kl=model_args.use_kl,
        predict_xstart=model_args.predict_xstart,
        rescale_timesteps=model_args.rescale_timesteps,
        rescale_learned_sigmas=model_args.rescale_learned_sigmas,
        use_checkpoint=model_args.use_checkpoint,
        use_scale_shift_norm=model_args.use_scale_shift_norm,
        resblock_updown=model_args.resblock_updown,
        use_fp16=model_args.use_fp16,
        use_new_attention_order=model_args.use_new_attention_order,
    )
    
    # Load weights
    print(f"   Loading weights from: {model_path}")
    state_dict = th.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    print("✓ Diffusion model loaded!")
    
    # Load classifier if using guidance
    classifier = None
    classifier_scale = config.get("classifier_scale", 1.0)
    if args.use_classifier and classifier_path and os.path.exists(classifier_path):
        print(f"\n📦 Loading classifier for guidance...")
        
        classifier = create_classifier(
            image_size=config["image_size"],
            classifier_use_fp16=args.use_fp16,
            classifier_width=config.get("classifier_width", 128),
            classifier_depth=config.get("classifier_depth", 2),
            classifier_attention_resolutions=config.get("classifier_attention_resolutions", "32,16,8"),
            classifier_use_scale_shift_norm=True,
            classifier_resblock_updown=True,
            classifier_pool="attention",
        )
        
        state_dict = th.load(classifier_path, map_location="cpu")
        classifier.load_state_dict(state_dict)
        classifier.to(device)
        if args.use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()
        print(f"✓ Classifier loaded! (guidance scale: {classifier_scale})")
    
    # Define classifier guidance function
    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * classifier_scale

    def model_fn(x, t, y=None):
        return model(x, t, y if config["class_cond"] else None)
    
    # Parse classes
    classes_list = None
    if args.classes:
        classes_list = [int(c.strip()) for c in args.classes.split(",")]
        print(f"\n🎯 Generating specific classes: {classes_list}")
    
    # Create output directory
    output_dir = os.path.join(SCRIPT_DIR, "outputs", f"generated_{args.resolution}x{args.resolution}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n🎨 Generating {args.num_samples} images at {args.resolution}x{args.resolution}...")
    print(f"   Steps: {args.steps}")
    print(f"   Mode: {'Classifier-guided' if args.use_classifier and classifier else 'Unconditional'}")
    print(f"   Output: {output_dir}")
    print()
    
    all_images = []
    all_labels = []
    
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    sample_idx = 0
    
    for batch_idx in range(num_batches):
        current_batch_size = min(args.batch_size, args.num_samples - batch_idx * args.batch_size)
        print(f"📊 Batch {batch_idx + 1}/{num_batches} ({current_batch_size} samples)...")
        
        model_kwargs = {}
        if config["class_cond"]:
            if classes_list:
                batch_classes = []
                for i in range(current_batch_size):
                    class_idx = (sample_idx + i) % len(classes_list)
                    batch_classes.append(classes_list[class_idx])
                classes = th.tensor(batch_classes, device=device)
                sample_idx += current_batch_size
            else:
                classes = th.randint(low=0, high=1000, size=(current_batch_size,), device=device)
            model_kwargs["y"] = classes
        
        # Sample
        with th.no_grad():
            if args.use_classifier and classifier:
                sample = diffusion.p_sample_loop(
                    model_fn,
                    (current_batch_size, 3, args.resolution, args.resolution),
                    clip_denoised=True,
                    model_kwargs=model_kwargs,
                    cond_fn=cond_fn,
                    device=device,
                    progress=True,
                )
            else:
                sample = diffusion.p_sample_loop(
                    model,
                    (current_batch_size, 3, args.resolution, args.resolution),
                    clip_denoised=True,
                    model_kwargs=model_kwargs,
                    progress=True,
                )
        
        # Convert to uint8
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        
        all_images.append(sample)
        if config["class_cond"]:
            all_labels.append(classes.cpu().numpy())
        
        print(f"   ✓ Batch {batch_idx + 1} complete")
    
    # Concatenate and save
    arr = np.concatenate(all_images, axis=0)[:args.num_samples]
    
    # Save as NPZ
    shape_str = "x".join([str(x) for x in arr.shape])
    npz_path = os.path.join(output_dir, f"samples_{shape_str}.npz")
    
    if all_labels:
        label_arr = np.concatenate(all_labels, axis=0)[:args.num_samples]
        np.savez(npz_path, arr, label_arr)
    else:
        np.savez(npz_path, arr)
    
    print(f"\n💾 Saved NPZ: {npz_path}")
    
    # Also save as individual PNG images
    print("\n📸 Saving PNG images...")
    for i, img in enumerate(arr):
        png_path = os.path.join(output_dir, f"sample_{i+1:03d}.png")
        Image.fromarray(img).save(png_path)
        print(f"   ✓ {png_path}")
    
    print(f"\n✅ Done! Generated {len(arr)} images")
    print(f"📁 Output directory: {output_dir}")
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Generate images using Guided Diffusion (Windows compatible)")
    parser.add_argument("--resolution", type=int, default=64, choices=[64, 128, 512],
                        help="Image resolution (64, 128, or 512)")
    parser.add_argument("--num_samples", type=int, default=4,
                        help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for generation")
    parser.add_argument("--steps", type=int, default=250,
                        help="Number of diffusion steps (fewer = faster, more = better quality)")
    parser.add_argument("--use_classifier", action="store_true",
                        help="Use classifier guidance for improved quality")
    parser.add_argument("--use_fp16", action="store_true",
                        help="Use half precision (faster, less VRAM)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--classes", type=str, default="",
                        help="Comma-separated ImageNet class IDs (e.g., '207,281,388')")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎨 GUIDED DIFFUSION - Image Generation")
    print("=" * 60)
    print()
    
    # Check for required models
    missing_models = check_models(args.resolution, args.use_classifier)
    if missing_models:
        print("❌ Missing required models:")
        for m in missing_models:
            print(f"   - models/{m}")
        print()
        print("📥 Download URLs:")
        for m in missing_models:
            print(f"   https://openaipublic.blob.core.windows.net/diffusion/jul-2021/{m}")
        print()
        print("Please download these models to the 'models/' folder and try again.")
        sys.exit(1)
    
    # Generate images
    generate_images(args)

if __name__ == "__main__":
    main()
