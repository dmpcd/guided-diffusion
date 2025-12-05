#!/bin/bash
# Generate 512x512 images of specific well-known ImageNet classes
# Based on the example image showing popular animals and objects

cd /home/senum/projects/guided-diffusion/guided-diffusion

echo "🎨 Generating 512×512 images of SPECIFIC classes"
echo "=================================================="
echo ""

# Well-known ImageNet classes (matching your reference image)
# Format: CLASS_ID = Class Name
#
# Row 1: Birds & Dogs
#   9   = Ostrich
#   130 = Flamingo  
#   263 = Pembroke Welsh Corgi
#   1   = Goldfish
#   207 = Golden Retriever
#   94  = Hummingbird
#
# Row 2: Wildlife
#   279 = Arctic Fox
#   289 = Snow Leopard
#   323 = Monarch Butterfly
#   367 = Chimpanzee  
#   388 = Giant Panda
#   387 = Red Panda (Lesser Panda)
#
# Row 3: Objects & Animals
#   852 = Tennis Ball
#   250 = Siberian Husky
#   609 = Pickup Truck (old style)
#   76  = Spider (Tarantula)
#   92  = Bee Eater (colorful bird)
#   928 = Ice Cream / Trifle

# Define the classes to generate
CLASSES="9,130,263,1,207,94,279,289,323,367,388,387,852,250,609,76,92,928"
NUM_SAMPLES=18

echo "Classes to generate:"
echo "  Row 1: Ostrich, Flamingo, Corgi, Goldfish, Golden Retriever, Hummingbird"
echo "  Row 2: Arctic Fox, Snow Leopard, Butterfly, Chimpanzee, Giant Panda, Red Panda"
echo "  Row 3: Tennis Ball, Husky, Pickup Truck, Spider, Bee Eater, Dessert"
echo ""
echo "Total: $NUM_SAMPLES images"
echo ""

# Create output directory
mkdir -p outputs/specific_classes

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
    --num_samples $NUM_SAMPLES \
    --batch_size 1 \
    --classes "$CLASSES" \
    --output_dir outputs/specific_classes

echo ""
echo "📸 Converting to PNG images with class names..."

python -c "
import numpy as np
from PIL import Image
import glob
import os

# Class ID to name mapping
CLASS_NAMES = {
    9: 'ostrich',
    130: 'flamingo',
    263: 'corgi',
    1: 'goldfish',
    207: 'golden_retriever',
    94: 'hummingbird',
    279: 'arctic_fox',
    289: 'snow_leopard',
    323: 'monarch_butterfly',
    367: 'chimpanzee',
    388: 'giant_panda',
    387: 'red_panda',
    852: 'tennis_ball',
    250: 'siberian_husky',
    609: 'pickup_truck',
    76: 'tarantula',
    92: 'bee_eater',
    928: 'trifle_dessert',
}

output_dir = 'outputs/specific_classes'
npz_files = glob.glob(f'{output_dir}/samples_*.npz')
latest_file = max(npz_files, key=os.path.getctime)

data = np.load(latest_file)
images = data['arr_0']
labels = data['arr_1'] if 'arr_1' in data.files else None

print(f'Loaded {len(images)} images')

for i, img in enumerate(images):
    if labels is not None:
        class_id = labels[i]
        class_name = CLASS_NAMES.get(class_id, f'class_{class_id}')
        save_path = f'{output_dir}/{i+1:02d}_{class_name}.png'
    else:
        save_path = f'{output_dir}/sample_{i+1:02d}.png'
    Image.fromarray(img).save(save_path)
    print(f'✓ {save_path}')
"

echo ""
echo "✅ Done! Check outputs/specific_classes/ for named PNG files"
