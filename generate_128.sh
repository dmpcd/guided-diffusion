#!/bin/bash
# Generate 128x128 base images for super-resolution upsampling

cd /home/senum/projects/guided-diffusion/guided-diffusion

echo "🎨 Generating 128×128 base images (with classifier guidance)..."
echo "   Purpose: Base images for 128→512 super-resolution"
echo "   Method: Classifier-guided generation"
echo ""

# Check if models exist
if [ ! -f "models/128x128_diffusion.pt" ]; then
    echo "⬇️  Downloading 128×128 diffusion model..."
    mkdir -p models
    wget -q --show-progress \
        https://openaipublic.blob.core.windows.net/diffusion/jul-2021/128x128_diffusion.pt \
        -O models/128x128_diffusion.pt
    echo ""
fi

if [ ! -f "models/128x128_classifier.pt" ]; then
    echo "⬇️  Downloading 128×128 classifier model..."
    mkdir -p models
    wget -q --show-progress \
        https://openaipublic.blob.core.windows.net/diffusion/jul-2021/128x128_classifier.pt \
        -O models/128x128_classifier.pt
    echo ""
fi

# Create output directory
mkdir -p outputs/base_128

# Model configuration for 128x128
MODEL_FLAGS="
    --attention_resolutions 32,16,8
    --class_cond True
    --diffusion_steps 1000
    --dropout 0.0
    --image_size 128
    --learn_sigma True
    --noise_schedule linear
    --num_channels 256
    --num_heads 4
    --num_res_blocks 2
    --resblock_updown True
    --use_scale_shift_norm True
"

# Classifier configuration
CLASSIFIER_FLAGS="
    --classifier_attention_resolutions 32,16,8
    --classifier_depth 2
    --classifier_width 128
    --classifier_pool attention
    --classifier_resblock_updown True
    --classifier_use_scale_shift_norm True
    --classifier_scale 4.0
"

# Generate samples with specific classes for comparison
# Classes: 270=arctic wolf, 283=Persian cat, 289=snow leopard, 290=jaguar, 291=lion,
#          294=brown bear, 296=polar bear, 301=ladybug, 334=porcupine, 72=spider
python simple_demo.py \
    $MODEL_FLAGS \
    $CLASSIFIER_FLAGS \
    --model_path models/128x128_diffusion.pt \
    --classifier_path models/128x128_classifier.pt \
    --timestep_respacing 250 \
    --num_samples 10 \
    --batch_size 10 \
    --use_fp16 True \
    --seed 42 \
    --classes "277,277,277,277,277,367,367,367,367,367" \
    --output_dir outputs/base_128

echo ""
echo "📸 Converting to PNG images..."

python -c "
import numpy as np
from PIL import Image
import glob
import os

output_dir = 'outputs/base_128'
npz_files = glob.glob(f'{output_dir}/samples_*.npz')
latest_file = max(npz_files, key=os.path.getctime)

data = np.load(latest_file)
images = data['arr_0']

for i, img in enumerate(images):
    save_path = f'{output_dir}/sample_{i+1:02d}.png'
    Image.fromarray(img).save(save_path)
    print(f'✓ {save_path}')
"

echo ""
echo "✅ Done! 128×128 base images generated"
echo "📁 Saved to: outputs/base_128/"
echo ""
echo "💡 Next step: Run ./upsample_128_to_512.sh to create 512×512 images"
