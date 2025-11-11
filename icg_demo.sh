#!/bin/bash
# Demo Script - ICG (Independent Condition Guidance)
# Generates samples WITHOUT needing a classifier model!

cd /home/senum/projects/guided-diffusion/guided-diffusion

echo "🎨 GUIDED DIFFUSION - ICG (Independent Condition Guidance)"
echo "=========================================================="
echo ""
echo "📋 Configuration:"
echo "   - Model: 64x64 ImageNet"
echo "   - Mode: ICG (NO classifier needed!)"
echo "   - ICG Scale: 1.5 (default)"
echo "   - Samples: 4 images"
echo "   - Steps: 250 (fast mode)"
echo "   - Output: outputs/icg/"
echo ""
echo "💡 ICG uses the diffusion model itself for guidance"
echo "   No classifier model required - training-free approach!"
echo ""

# Create output directory
mkdir -p outputs/icg

echo "🚀 Starting generation with ICG..."
echo ""

python icg_demo.py \
    --model_path models/64x64_diffusion.pt \
    --attention_resolutions 32,16,8 \
    --class_cond True \
    --diffusion_steps 1000 \
    --dropout 0.1 \
    --image_size 64 \
    --learn_sigma True \
    --noise_schedule cosine \
    --num_channels 192 \
    --num_head_channels 64 \
    --num_res_blocks 3 \
    --resblock_updown True \
    --use_new_attention_order True \
    --use_scale_shift_norm True \
    --timestep_respacing 250 \
    --num_samples 4 \
    --batch_size 2 \
    --seed 55 \
    --icg_scale 1.5 \
    --output_dir outputs/icg

echo ""
echo "✅ Generation complete!"
echo ""
echo "📁 Viewing results..."

# View results
python -c "
import numpy as np
from PIL import Image
import os
import glob

output_dir = 'outputs/icg'
npz_files = glob.glob(f'{output_dir}/samples_*.npz')
if not npz_files:
    print('❌ No samples found!')
    exit(1)

latest_file = max(npz_files, key=os.path.getctime)
print(f'\n📂 Loading: {latest_file}')

data = np.load(latest_file)
images = data['arr_0']
labels = data['arr_1'] if 'arr_1' in data else None
print(f'✓ Loaded {len(images)} images')

if labels is not None:
    print(f'   Generated classes: {labels.tolist()}')

os.makedirs(output_dir, exist_ok=True)

for i, img in enumerate(images):
    save_path = f'{output_dir}/sample_{i+1:02d}.png'
    Image.fromarray(img).save(save_path)
    print(f'   ✓ {save_path}')

print(f'\n✅ Results saved to: {output_dir}/')
"

echo ""
echo "=========================================================="
echo "✅ DONE! Check outputs/icg/"
echo ""
echo "💡 Key Advantage of ICG:"
echo "   - No classifier model needed (saves ~250MB)"
echo "   - Training-free approach"
echo "   - Similar quality to classifier guidance"
echo "   - Faster inference (no classifier gradients)"
echo ""
echo "To compare with other methods:"
echo "   - Without guidance: ./without_classifier.sh"
echo "   - With classifier:  ./with_classifier_guidance.sh"
echo "   - With ICG:        ./icg_demo.sh"
echo "=========================================================="
