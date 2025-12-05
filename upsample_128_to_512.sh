#!/bin/bash
# Upsample 128x128 images to 512x512 using super-resolution diffusion model

cd /home/senum/projects/guided-diffusion/guided-diffusion

echo "📈 Super-Resolution: 128×128 → 512×512"
echo "   Method: Two-stage upsampling"
echo "   Quality: BEST (FID ~3.85)"
echo ""

# Check if base samples exist
BASE_SAMPLES=${1:-"outputs/base_128/samples_10x128x128x3.npz"}

if [ ! -f "$BASE_SAMPLES" ]; then
    echo "❌ Error: Base samples not found at $BASE_SAMPLES"
    echo ""
    echo "Usage: $0 [path_to_128x128_samples.npz]"
    echo ""
    echo "Please generate 128×128 base samples first:"
    echo "  ./generate_128.sh"
    echo ""
    exit 1
fi

echo "📂 Using base samples: $BASE_SAMPLES"
echo ""

# Create output directory
mkdir -p outputs/super_res

# Model configuration for 128→512 upsampler
MODEL_FLAGS="
    --attention_resolutions 32,16
    --class_cond True
    --diffusion_steps 1000
    --large_size 512
    --small_size 128
    --learn_sigma True
    --noise_schedule linear
    --num_channels 192
    --num_head_channels 64
    --num_res_blocks 2
    --resblock_updown True
    --use_scale_shift_norm True
"

# Check if upsampler model exists
UPSAMPLER_MODEL="models/128_512_upsampler.pt"

if [ ! -f "$UPSAMPLER_MODEL" ]; then
    echo "⬇️  Downloading 128→512 upsampler model..."
    echo ""
    mkdir -p models
    wget -q --show-progress \
        https://openaipublic.blob.core.windows.net/diffusion/jul-2021/128_512_upsampler.pt \
        -O "$UPSAMPLER_MODEL"
    echo ""
fi

# Run super-resolution with specific classes for comparison
# Classes: 270=arctic wolf, 283=Persian cat, 289=snow leopard, 290=jaguar, 291=lion,
#          294=brown bear, 296=polar bear, 301=ladybug, 334=porcupine, 72=spider
python super_res_demo.py \
    $MODEL_FLAGS \
    --model_path "$UPSAMPLER_MODEL" \
    --base_samples "$BASE_SAMPLES" \
    --num_samples 10 \
    --batch_size 1 \
    --timestep_respacing 250 \
    --use_fp16 True \
    --output_dir outputs/super_res \
    --save_png True \
    --seed 42 \
    --class_idx "277,277,277,277,277,367,367,367,367,367"

echo ""
echo "✅ Super-resolution complete!"
echo "📁 Check outputs/super_res/ for results"
