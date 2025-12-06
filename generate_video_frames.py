#!/usr/bin/env python3
"""
Generate video frames showing the diffusion denoising process.
Saves intermediate images at each timestep to create a visualization video.

Usage:
    python generate_video_frames.py \
        --model_path models/512x512_diffusion.pt \
        --classifier_path models/512x512_classifier.pt \
        --image_size 512 \
        --class_id 207 \
        --output_dir outputs/video_frames \
        --save_every 1

This will save frames at each denoising step, which you can then combine into a video.
"""
import argparse
import os
import sys

sys.path.insert(0, '/home/senum/projects/guided-diffusion/guided-diffusion')

import numpy as np
import torch as th
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)

# ImageNet class names for common classes
IMAGENET_CLASSES = {
    9: "ostrich",
    130: "flamingo",
    207: "golden_retriever",
    250: "husky",
    263: "corgi",
    281: "tabby_cat",
    291: "lion",
    323: "monarch_butterfly",
    388: "panda",
    417: "balloon",
    562: "fountain",
    852: "tennis_ball",
    928: "ice_cream",
    933: "cheeseburger",
    949: "strawberry",
    980: "volcano",
    985: "daisy",
}


def get_class_name(class_id):
    """Get human-readable class name."""
    return IMAGENET_CLASSES.get(class_id, f"class_{class_id}")


def tensor_to_image(tensor):
    """Convert a tensor to a PIL Image."""
    # tensor shape: (C, H, W) in range [-1, 1]
    img = ((tensor + 1) * 127.5).clamp(0, 255).to(th.uint8)
    img = img.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
    return Image.fromarray(img)


def main():
    args = create_argparser().parse_args()
    
    # Set random seed for reproducibility
    if args.seed is not None:
        th.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"🎲 Random seed set to: {args.seed}")
    
    print("=" * 60)
    print("    DIFFUSION VIDEO FRAME GENERATOR")
    print("=" * 60)
    print(f"\n🎯 Target class: {args.class_id} ({get_class_name(args.class_id)})")
    print(f"📐 Resolution: {args.image_size}×{args.image_size}")
    print(f"💾 Output directory: {args.output_dir}")
    print(f"📸 Save every {args.save_every} step(s)")
    
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("\n📦 Loading diffusion model...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    state_dict = th.load(args.model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    print("✓ Diffusion model loaded!")
    
    # Load classifier
    print("\n📦 Loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    state_dict = th.load(args.classifier_path, map_location="cpu", weights_only=True)
    classifier.load_state_dict(state_dict)
    classifier.to(device)
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()
    print(f"✓ Classifier loaded! (guidance scale: {args.classifier_scale})")
    
    # Get number of timesteps
    num_timesteps = diffusion.num_timesteps
    print(f"\n⏱️  Total timesteps: {num_timesteps}")
    
    # Classifier guidance function
    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        return model(x, t, y if args.class_cond else None)

    # Prepare class label
    classes = th.tensor([args.class_id], device=device)
    model_kwargs = {"y": classes}
    
    # Initialize with noise
    shape = (1, 3, args.image_size, args.image_size)
    noise = th.randn(*shape, device=device)
    
    # Save initial noise image
    print("\n🎬 Starting generation and saving frames...")
    noise_img = tensor_to_image(noise[0])
    noise_path = os.path.join(args.output_dir, "frame_0000_noise.png")
    noise_img.save(noise_path)
    print(f"   Saved: frame_0000_noise.png (pure noise)")
    
    # Generate with progressive sampling
    frames_saved = 1
    all_frames = [noise_img]
    
    indices = list(range(num_timesteps))[::-1]  # T-1, T-2, ..., 0
    img = noise
    
    for step_idx, i in enumerate(tqdm(indices, desc="Denoising")):
        t = th.tensor([i], device=device)
        
        with th.no_grad():
            # Perform one denoising step
            out = diffusion.p_sample(
                model_fn,
                img,
                t,
                clip_denoised=args.clip_denoised,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
            )
            img = out["sample"]
        
        # Save frame at specified intervals
        if (step_idx + 1) % args.save_every == 0 or step_idx == len(indices) - 1:
            frame_img = tensor_to_image(img[0])
            frame_num = str(frames_saved).zfill(4)
            timestep_str = str(i).zfill(4)
            frame_path = os.path.join(args.output_dir, f"frame_{frame_num}_t{timestep_str}.png")
            frame_img.save(frame_path)
            all_frames.append(frame_img)
            frames_saved += 1
    
    # Save final image with special name
    final_img = tensor_to_image(img[0])
    final_path = os.path.join(args.output_dir, f"final_{get_class_name(args.class_id)}.png")
    final_img.save(final_path)
    
    print(f"\n✅ Done! Saved {frames_saved} frames to {args.output_dir}")
    print(f"📁 Final image: {final_path}")
    
    # Create a simple GIF if requested
    if args.create_gif:
        gif_path = os.path.join(args.output_dir, f"diffusion_{get_class_name(args.class_id)}.gif")
        print(f"\n🎥 Creating GIF: {gif_path}")
        
        # Use every Nth frame for GIF to keep file size reasonable
        gif_frames = all_frames[::max(1, len(all_frames) // 50)]  # Max ~50 frames in GIF
        
        # Add final frame multiple times to pause at the end
        gif_frames.extend([all_frames[-1]] * 10)
        
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=100,  # ms per frame
            loop=0
        )
        print(f"✓ GIF saved: {gif_path}")
    
    # Print instructions for creating video with ffmpeg
    print("\n" + "=" * 60)
    print("📹 TO CREATE A VIDEO, run this command:")
    print("=" * 60)
    print(f"""
ffmpeg -framerate 30 -pattern_type glob -i '{args.output_dir}/frame_*.png' \\
    -c:v libx264 -pix_fmt yuv420p -crf 18 \\
    {args.output_dir}/diffusion_process.mp4

Or for a slower video (10 fps):
ffmpeg -framerate 10 -pattern_type glob -i '{args.output_dir}/frame_*.png' \\
    -c:v libx264 -pix_fmt yuv420p -crf 18 \\
    {args.output_dir}/diffusion_process_slow.mp4
""")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        use_ddim=False,
        model_path="models/512x512_diffusion.pt",
        classifier_path="models/512x512_classifier.pt",
        classifier_scale=4.0,
        output_dir="outputs/video_frames",
        seed=42,
        class_id=207,  # golden retriever by default
        save_every=1,  # Save every N steps (1 = all steps)
        create_gif=True,  # Also create an animated GIF
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
