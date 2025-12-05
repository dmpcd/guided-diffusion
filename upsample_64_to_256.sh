#!/bin/bash
# Upsample 64x64 images to 256x256 using super-resolution diffusion model

cd /home/senum/projects/guided-diffusion/guided-diffusion

echo "📈 Super-Resolution: 64×64 → 256×256"
echo ""

# Check if base samples exist
BASE_SAMPLES=${1:-"outputs/with_classifier/samples_4x64x64x3.npz"}

if [ ! -f "$BASE_SAMPLES" ]; then
    echo "❌ Error: Base samples not found at $BASE_SAMPLES"
    echo ""
    echo "Usage: $0 [path_to_64x64_samples.npz]"
    echo ""
    echo "Please generate 64x64 samples first using:"
    echo "  ./with_classifier_guidance.sh"
    echo "  or"
    echo "  ./without_classifier.sh"
    exit 1
fi

echo "📂 Using base samples: $BASE_SAMPLES"
echo ""

# Create output directory
mkdir -p outputs/super_res

# Model configuration for 64→256 upsampler
MODEL_FLAGS="
    --attention_resolutions 32,16,8
    --class_cond True
    --diffusion_steps 1000
    --large_size 256
    --small_size 64
    --learn_sigma True
    --noise_schedule linear
    --num_channels 192
    --num_heads 4
    --num_res_blocks 2
    --resblock_updown True
    --use_scale_shift_norm True
"

# Check if upsampler model exists
UPSAMPLER_MODEL="models/64_256_upsampler.pt"

if [ ! -f "$UPSAMPLER_MODEL" ]; then
    echo "⬇️  Downloading 64→256 upsampler model..."
    echo ""
    mkdir -p models
    wget -q --show-progress \
        https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64_256_upsampler.pt \
        -O "$UPSAMPLER_MODEL"
    echo ""
fi

# Run super-resolution with specific classes for comparison
# Classes: 99=Goose, 1=Goldfish, 949=Strawberry, 417=Balloon
python super_res_demo.py \
    $MODEL_FLAGS \
    --model_path "$UPSAMPLER_MODEL" \
    --base_samples "$BASE_SAMPLES" \
    --num_samples 4 \
    --batch_size 2 \
    --timestep_respacing 250 \
    --use_fp16 True \
    --output_dir outputs/super_res \
    --save_png True \
    --seed 42 \
    --class_idx "99,1,949,417"

echo ""
echo "✅ Super-resolution complete!"
echo "📁 Check outputs/super_res/ for results"
