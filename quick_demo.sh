#!/bin/bash
# Quick Demo Script for Guided Diffusion
# Run this during your presentation

cd /home/senum/projects/guided-diffusion/guided-diffusion

echo "🎨 GUIDED DIFFUSION - LIVE DEMO"
echo "================================"
echo ""
echo "📋 Configuration:"
echo "   - Model: 64x64 ImageNet"
echo "   - Samples: 4 images"
echo "   - Steps: 250 (fast mode)"
echo "   - Hardware: RTX 4090"
echo ""
echo "🚀 Starting generation..."
echo ""

python simple_demo.py \
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
    --batch_size 4

echo ""
echo "✅ Generation complete!"
echo ""
echo "📁 Viewing results..."
python demo_view_results.py
