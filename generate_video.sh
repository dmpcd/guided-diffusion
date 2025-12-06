#!/bin/bash
# Generate video frames showing the diffusion denoising process
# This creates intermediate images at each timestep for video creation

cd /home/senum/projects/guided-diffusion/guided-diffusion

# Configuration
CLASS_ID=${1:-207}  # Default: golden retriever (207)
SAVE_EVERY=${2:-1}  # Save every N steps (1 = all 250 steps)

# Class name lookup for output folder
declare -A CLASS_NAMES
CLASS_NAMES[9]="ostrich"
CLASS_NAMES[130]="flamingo"
CLASS_NAMES[207]="golden_retriever"
CLASS_NAMES[250]="husky"
CLASS_NAMES[263]="corgi"
CLASS_NAMES[281]="tabby_cat"
CLASS_NAMES[291]="lion"
CLASS_NAMES[323]="monarch_butterfly"
CLASS_NAMES[388]="panda"
CLASS_NAMES[417]="balloon"
CLASS_NAMES[933]="cheeseburger"
CLASS_NAMES[949]="strawberry"
CLASS_NAMES[985]="daisy"

CLASS_NAME=${CLASS_NAMES[$CLASS_ID]:-"class_$CLASS_ID"}
OUTPUT_DIR="outputs/video_frames/${CLASS_NAME}"

echo "=============================================="
echo "  DIFFUSION VIDEO FRAME GENERATOR (512x512)"
echo "=============================================="
echo ""
echo "Class ID: $CLASS_ID ($CLASS_NAME)"
echo "Save every: $SAVE_EVERY steps"
echo "Output: $OUTPUT_DIR"
echo ""

python generate_video_frames.py \
    --model_path models/512x512_diffusion.pt \
    --classifier_path models/512x512_classifier.pt \
    --image_size 512 \
    --attention_resolutions 32,16,8 \
    --class_cond True \
    --diffusion_steps 1000 \
    --timestep_respacing 250 \
    --learn_sigma True \
    --noise_schedule linear \
    --num_channels 256 \
    --num_head_channels 64 \
    --num_res_blocks 2 \
    --resblock_updown True \
    --use_fp16 True \
    --use_scale_shift_norm True \
    --classifier_scale 4.0 \
    --classifier_attention_resolutions 32,16,8 \
    --classifier_depth 2 \
    --classifier_width 128 \
    --classifier_pool attention \
    --classifier_resblock_updown True \
    --classifier_use_scale_shift_norm True \
    --classifier_use_fp16 True \
    --class_id $CLASS_ID \
    --output_dir "$OUTPUT_DIR" \
    --save_every $SAVE_EVERY \
    --create_gif True \
    --seed 42

echo ""
echo "=============================================="
echo "  DONE!"
echo "=============================================="
echo ""
echo "Frames saved to: $OUTPUT_DIR"
echo ""
echo "To create a video, run:"
echo "  ffmpeg -framerate 30 -pattern_type glob -i '${OUTPUT_DIR}/frame_*.png' \\"
echo "      -c:v libx264 -pix_fmt yuv420p ${OUTPUT_DIR}/diffusion_video.mp4"
