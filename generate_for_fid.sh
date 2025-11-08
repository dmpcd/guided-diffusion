#!/bin/bash
# Generate larger sample sets for FID evaluation
# Recommended: 2000+ samples for meaningful FID scores

cd /home/senum/projects/guided-diffusion/guided-diffusion

NUM_SAMPLES=${1:-100}  # Default 100 samples, or use first argument
BATCH_SIZE=${2:-10}     # Default batch size 10

echo "🎨 GENERATING SAMPLES FOR FID EVALUATION"
echo "========================================="
echo ""
echo "📋 Configuration:"
echo "   - Samples: $NUM_SAMPLES"
echo "   - Batch size: $BATCH_SIZE"
echo "   - Resolution: 64x64"
echo ""

# Create output directories
mkdir -p outputs/fid_evaluation/without_classifier
mkdir -p outputs/fid_evaluation/with_classifier

echo "======================="
echo "📊 Step 1/2: WITHOUT Classifier"
echo "======================="
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
    --num_samples $NUM_SAMPLES \
    --batch_size $BATCH_SIZE \
    --output_dir outputs/fid_evaluation/without_classifier

echo ""
echo "======================="
echo "📊 Step 2/2: WITH Classifier"
echo "======================="
echo ""

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
    --num_samples $NUM_SAMPLES \
    --batch_size $BATCH_SIZE \
    --output_dir outputs/fid_evaluation/with_classifier

echo ""
echo "========================================="
echo "✅ Sample generation complete!"
echo ""
echo "Generated samples:"
echo "  - outputs/fid_evaluation/without_classifier/samples_*.npz"
echo "  - outputs/fid_evaluation/with_classifier/samples_*.npz"
echo ""
echo "Next step: Run FID evaluation"
echo "  ./evaluate_fid.sh"
echo "========================================="
