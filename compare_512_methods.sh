#!/bin/bash
# Compare direct 512×512 generation vs 128→512 super-resolution
# This script generates both and allows visual/quantitative comparison

cd /home/senum/projects/guided-diffusion/guided-diffusion

echo "═══════════════════════════════════════════════════════════════"
echo "  512×512 Generation Quality Comparison"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "This script will generate 512×512 images using two methods:"
echo ""
echo "  Method 1: Direct 512×512 generation"
echo "            - Single-stage"
echo "            - Faster (~60s per image)"
echo "            - FID: ~7.72"
echo ""
echo "  Method 2: 128→512 super-resolution"
echo "            - Two-stage (128×128 → 512×512)"
echo "            - Slower (~75s per image)"
echo "            - FID: ~3.85 (2× BETTER!)"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""

read -p "Generate both for comparison? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STAGE 1/3: Direct 512×512 Generation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

./generate_512.sh

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STAGE 2/3: Generate 128×128 Base Images"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

./generate_128.sh

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STAGE 3/3: Upsample 128×128 → 512×512"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

./upsample_128_to_512.sh

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✅ Comparison Complete!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Results are saved in:"
echo ""
echo "  📁 outputs/direct_512/        - Direct generation (FID ~7.72)"
echo "  📁 outputs/base_128/          - 128×128 base images"
echo "  📁 outputs/super_res/         - Super-resolution (FID ~3.85)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Visual Comparison"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Compare the images:"
echo ""
echo "  Direct:        outputs/direct_512/sample_01.png"
echo "  Super-res:     outputs/super_res/upsampled_512x512/upsampled_0001.png"
echo ""
echo "Expected differences:"
echo "  ✓ Super-res should have sharper details"
echo "  ✓ Super-res should have more realistic textures"
echo "  ✓ Super-res faces should be more coherent"
echo "  ✓ Direct generation may look softer/blurrier"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Optional: Create side-by-side comparison
read -p "Create side-by-side comparison images? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Creating side-by-side comparisons..."
    
    python -c "
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# Create comparison directory
os.makedirs('outputs/comparison_512', exist_ok=True)

# Load images
direct_files = sorted([f'outputs/direct_512/sample_{i:02d}.png' for i in range(1, 11)])
upsampled_files = sorted([f'outputs/super_res/upsampled_512x512/upsampled_{i:04d}.png' for i in range(1, 11)])

for i, (direct_path, upsampled_path) in enumerate(zip(direct_files, upsampled_files), 1):
    if not os.path.exists(direct_path) or not os.path.exists(upsampled_path):
        continue
    
    # Load images
    direct = Image.open(direct_path)
    upsampled = Image.open(upsampled_path)
    
    # Create side-by-side
    width, height = direct.size
    comparison = Image.new('RGB', (width * 2 + 40, height + 80), 'white')
    
    # Add images
    comparison.paste(direct, (20, 60))
    comparison.paste(upsampled, (width + 20, 60))
    
    # Add labels
    draw = ImageDraw.Draw(comparison)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 24)
    except:
        font = ImageFont.load_default()
    
    # Title
    draw.text((width, 20), f'Sample {i} Comparison', fill='black', font=font, anchor='mt')
    
    # Method labels
    draw.text((width // 2 + 20, height + 65), 'Direct 512x512', fill='black', font=font, anchor='mt')
    draw.text((width + width // 2 + 20, height + 65), 'Super-Resolution', fill='green', font=font, anchor='mt')
    
    # Save
    save_path = f'outputs/comparison_512/comparison_{i:02d}.png'
    comparison.save(save_path)
    print(f'✓ Created {save_path}')

print('')
print('Side-by-side comparisons saved to outputs/comparison_512/')
"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  📊 Summary"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "According to the paper (Diffusion Models Beat GANs):"
echo ""
echo "  Direct 512×512:           FID = 7.72"
echo "  128→512 Super-Resolution: FID = 3.85 (2× better!)"
echo ""
echo "The super-resolution approach produces significantly"
echo "better quality at only ~25% increase in generation time."
echo ""
echo "✅ All done!"
