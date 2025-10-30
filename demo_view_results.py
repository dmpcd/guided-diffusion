#!/usr/bin/env python3
"""
Quick viewer for generated samples
"""
import numpy as np
from PIL import Image
import os
import glob

print("\n🖼️  VIEWING GENERATED SAMPLES")
print("=" * 50)

# Find the most recent samples file
npz_files = glob.glob('samples_*.npz')
if not npz_files:
    print("❌ No samples found! Did generation complete?")
    exit(1)

latest_file = max(npz_files, key=os.path.getctime)
print(f"\n📂 Loading: {latest_file}")

# Load samples
data = np.load(latest_file)
images = data['arr_0']
print(f"✓ Loaded {len(images)} images")
print(f"✓ Image shape: {images[0].shape}")
print(f"✓ Value range: [{images.min()}, {images.max()}]")

# Create output directory
os.makedirs('demo_output', exist_ok=True)

# Save individual images
print(f"\n💾 Saving individual images...")
for i, img in enumerate(images):
    save_path = f'demo_output/sample_{i+1:02d}.png'
    Image.fromarray(img).save(save_path)
    print(f"   ✓ {save_path}")

# Create a grid visualization
try:
    import matplotlib.pyplot as plt
    
    print(f"\n🎨 Creating grid visualization...")
    
    # Determine grid size
    n = len(images)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flat if rows > 1 else axes
    
    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i])
            ax.set_title(f'Sample {i+1}', fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    grid_path = 'demo_output/all_samples_grid.png'
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ {grid_path}")
    plt.close()
    
except ImportError:
    print("\n⚠️  matplotlib not available - skipping grid")

print("\n" + "=" * 50)
print("✅ COMPLETE!")
print(f"\n📁 All outputs saved to: demo_output/")
print(f"\nTo view:")
print(f"  - Individual images: demo_output/sample_*.png")
print(f"  - Grid view: demo_output/all_samples_grid.png")
print("\n" + "=" * 50 + "\n")
