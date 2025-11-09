#!/bin/bash
# Demo Script - WITH Classifier Guidance
# Generates improved samples using classifier gradients

cd /home/senum/projects/guided-diffusion/guided-diffusion

echo "🎨 GUIDED DIFFUSION - WITH CLASSIFIER GUIDANCE"
echo "=============================================="
echo ""
echo "📋 Configuration:"
echo "   - Model: 64x64 ImageNet"
echo "   - Mode: Classifier-guided (scale=1.0)"
echo "   - Samples: 4 images"
echo "   - Steps: 250 (fast mode)"
echo "   - Output: outputs/with_classifier/"
echo ""

# Check if classifier exists
if [ ! -f "models/64x64_classifier.pt" ]; then
    echo "❌ Classifier model not found!"
    echo ""
    echo "Please download it first:"
    echo "  cd models"
    echo "  wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt"
    echo "  cd .."
    exit 1
fi

echo "🚀 Starting generation with classifier guidance..."
echo ""

# Create output directory
mkdir -p outputs/with_classifier

python simple_demo.py \
    --model_path models/64x64_diffusion.pt \
    --classifier_path models/64x64_classifier.pt \
    --classifier_scale 1.0 \
    --classifier_depth 4 \
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
    --output_dir outputs/with_classifier

echo ""
echo "✅ Generation complete!"
echo ""
echo "📁 Viewing results..."

# View results from the specific output directory
python -c "
import numpy as np
from PIL import Image
import os
import glob

output_dir = 'outputs/with_classifier'
npz_files = glob.glob(f'{output_dir}/samples_*.npz')
if not npz_files:
    print('❌ No samples found!')
    exit(1)

latest_file = max(npz_files, key=os.path.getctime)
print(f'\n📂 Loading: {latest_file}')

data = np.load(latest_file)
images = data['arr_0']
print(f'✓ Loaded {len(images)} images')

os.makedirs(output_dir, exist_ok=True)

for i, img in enumerate(images):
    save_path = f'{output_dir}/sample_{i+1:02d}.png'
    Image.fromarray(img).save(save_path)
    print(f'   ✓ {save_path}')

print(f'\n✅ Results saved to: {output_dir}/')
print(f'📊 Compare with baseline using: diff outputs/without_classifier/ outputs/with_classifier/')
"

echo ""
echo "=============================================="
echo "✅ DONE! Check outputs/with_classifier/"
echo ""
echo "To compare results:"
echo "  ls -l outputs/without_classifier/"
echo "  ls -l outputs/with_classifier/"
echo "=============================================="
