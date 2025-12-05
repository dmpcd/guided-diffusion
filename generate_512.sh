#!/bin/bash
# Generate 512x512 images with classifier guidance

cd /home/senum/projects/guided-diffusion/guided-diffusion

echo "🎨 Generating 512×512 images directly (with classifier guidance)..."
echo "   Method: Direct generation"
echo "   Quality: Good (FID ~7.72)"
echo ""

# Create output directory
mkdir -p outputs/direct_512

# Generate same classes for fair comparison
# Classes: 270=arctic wolf, 283=Persian cat, 289=snow leopard, 290=jaguar, 291=lion,
#          294=brown bear, 296=polar bear, 301=ladybug, 334=porcupine, 72=spider
python simple_demo.py \
    --model_path models/512x512_diffusion.pt \
    --classifier_path models/512x512_classifier.pt \
    --classifier_scale 4.0 \
    --classifier_width 128 \
    --classifier_depth 2 \
    --classifier_attention_resolutions 32,16,8 \
    --attention_resolutions 32,16,8 \
    --class_cond True \
    --diffusion_steps 1000 \
    --dropout 0.0 \
    --image_size 512 \
    --learn_sigma True \
    --noise_schedule linear \
    --num_channels 256 \
    --num_head_channels 64 \
    --num_res_blocks 2 \
    --resblock_updown True \
    --use_new_attention_order False \
    --use_scale_shift_norm True \
    --timestep_respacing 250 \
    --num_samples 10 \
    --batch_size 1 \
    --seed 42 \
    --classes "270,283,289,290,291,294,296,301,334,72" \
    --output_dir outputs/direct_512

echo ""
echo "📸 Converting to PNG images..."

python -c "
import numpy as np
from PIL import Image
import glob
import os

output_dir = 'outputs/direct_512'
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
echo "✅ Done! Direct 512×512 generation complete"
echo "📁 Check outputs/direct_512/ for PNG files"
