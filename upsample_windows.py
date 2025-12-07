#!/usr/bin/env python3
"""
Windows-compatible super-resolution upsampling script
Upsamples 128x128 images to 512x512 using diffusion-based super-resolution

Usage:
    Step 1: Generate 128x128 base images
        python generate_images_windows.py --resolution 128 --num_samples 4 --use_classifier --classes "1,207,323,92"
    
    Step 2: Upsample to 512x512
        python upsample_windows.py --base_samples outputs/generated_128x128/samples_4x128x128x3.npz
"""
import argparse
import os
import sys
import glob

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import torch as th
from PIL import Image

from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
)


def main():
    parser = argparse.ArgumentParser(description="Upsample 128x128 images to 512x512")
    parser.add_argument("--base_samples", type=str, required=True,
                        help="Path to the 128x128 samples NPZ file")
    parser.add_argument("--output_dir", type=str, default="outputs/upsampled_512",
                        help="Output directory for upsampled images")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (use 1 for RTX 3050 4GB)")
    parser.add_argument("--steps", type=int, default=250,
                        help="Number of diffusion steps")
    parser.add_argument("--use_fp16", action="store_true",
                        help="Use half precision (faster, less VRAM)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("📈 SUPER-RESOLUTION: 128×128 → 512×512")
    print("=" * 60)
    print()
    
    # Check base samples exist
    if not os.path.exists(args.base_samples):
        print(f"❌ Error: Base samples not found at {args.base_samples}")
        print()
        print("Please generate 128x128 base samples first:")
        print('  python generate_images_windows.py --resolution 128 --num_samples 4 --use_classifier --classes "1,207,323,92"')
        sys.exit(1)
    
    # Check upsampler model exists
    upsampler_path = os.path.join(SCRIPT_DIR, "models", "128_512_upsampler.pt")
    if not os.path.exists(upsampler_path):
        print(f"❌ Error: Upsampler model not found at {upsampler_path}")
        print()
        print("Please download it:")
        print("  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/128_512_upsampler.pt")
        sys.exit(1)
    
    # Set device
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    if device.type == 'cpu':
        print("⚠️  Warning: Running on CPU will be very slow!")
    
    # Set random seed
    if args.seed is not None:
        th.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"🎲 Random seed: {args.seed}")
    
    # Load base samples
    print(f"\n📂 Loading base samples from: {args.base_samples}")
    data = np.load(args.base_samples)
    base_images = data['arr_0']
    base_labels = data['arr_1'] if 'arr_1' in data.files else None
    
    num_samples = len(base_images)
    print(f"   Found {num_samples} images at {base_images.shape[1]}x{base_images.shape[2]}")
    
    if base_labels is not None:
        print(f"   Labels: {base_labels.tolist()}")
    
    # Create model
    print(f"\n📦 Loading 128→512 upsampler model...")
    
    model, diffusion = sr_create_model_and_diffusion(
        large_size=512,
        small_size=128,
        class_cond=True,
        learn_sigma=True,
        num_channels=192,
        num_res_blocks=2,
        num_heads=-1,
        num_head_channels=64,
        num_heads_upsample=-1,
        attention_resolutions="32,16",
        dropout=0.0,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing=str(args.steps),
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=args.use_fp16,
    )
    
    # Load weights
    print(f"   Loading weights from: {upsampler_path}")
    state_dict = th.load(upsampler_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    print("✓ Upsampler model loaded!")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n🎨 Upsampling {num_samples} images from 128×128 to 512×512...")
    print(f"   Steps: {args.steps}")
    print(f"   Output: {args.output_dir}")
    print()
    
    all_upsampled = []
    
    for i in range(0, num_samples, args.batch_size):
        batch_end = min(i + args.batch_size, num_samples)
        current_batch_size = batch_end - i
        
        print(f"📊 Processing image {i+1}-{batch_end}/{num_samples}...")
        
        # Prepare low-res batch
        low_res_batch = base_images[i:batch_end]
        low_res_batch = th.from_numpy(low_res_batch).float()
        low_res_batch = low_res_batch / 127.5 - 1.0  # Normalize to [-1, 1]
        low_res_batch = low_res_batch.permute(0, 3, 1, 2)  # BHWC -> BCHW
        low_res_batch = low_res_batch.to(device)
        
        model_kwargs = {"low_res": low_res_batch}
        
        # Add class labels if available
        if base_labels is not None:
            labels = th.from_numpy(base_labels[i:batch_end]).to(device)
            model_kwargs["y"] = labels
        
        # Upsample
        with th.no_grad():
            upsampled = diffusion.p_sample_loop(
                model,
                (current_batch_size, 3, 512, 512),
                clip_denoised=True,
                model_kwargs=model_kwargs,
                progress=True,
            )
        
        # Convert to uint8
        upsampled = ((upsampled + 1) * 127.5).clamp(0, 255).to(th.uint8)
        upsampled = upsampled.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        
        all_upsampled.append(upsampled)
        print(f"   ✓ Batch complete")
    
    # Concatenate all results
    arr = np.concatenate(all_upsampled, axis=0)
    
    # Save as NPZ
    shape_str = "x".join([str(x) for x in arr.shape])
    npz_path = os.path.join(args.output_dir, f"upsampled_{shape_str}.npz")
    
    if base_labels is not None:
        np.savez(npz_path, arr, base_labels)
    else:
        np.savez(npz_path, arr)
    
    print(f"\n💾 Saved NPZ: {npz_path}")
    
    # Save as individual PNG images
    print("\n📸 Saving PNG images...")
    
    # Class name mapping for your classes
    CLASS_NAMES = {
        1: "goldfish",
        207: "golden_retriever", 
        323: "monarch_butterfly",
        92: "bee_eater",
    }
    
    for i, img in enumerate(arr):
        if base_labels is not None:
            class_id = base_labels[i]
            class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
            png_path = os.path.join(args.output_dir, f"{i+1:02d}_{class_name}_512x512.png")
        else:
            png_path = os.path.join(args.output_dir, f"upsampled_{i+1:03d}.png")
        
        Image.fromarray(img).save(png_path)
        print(f"   ✓ {png_path}")
    
    # Also save the original 128x128 for comparison
    print("\n📸 Saving original 128x128 images for comparison...")
    for i, img in enumerate(base_images):
        if base_labels is not None:
            class_id = base_labels[i]
            class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
            png_path = os.path.join(args.output_dir, f"{i+1:02d}_{class_name}_128x128_original.png")
        else:
            png_path = os.path.join(args.output_dir, f"original_{i+1:03d}_128x128.png")
        
        Image.fromarray(img).save(png_path)
        print(f"   ✓ {png_path}")
    
    print(f"\n✅ Done! Upsampled {len(arr)} images from 128×128 to 512×512")
    print(f"📁 Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
