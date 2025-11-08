# 🎯 PRESENTATION CHEAT SHEET
## Quick Reference for Your Demo

---

## ⚡ BEFORE PRESENTATION (5 minutes before)

```bash
cd /home/senum/projects/guided-diffusion/guided-diffusion
./setup_demo.sh
```

This will:
- ✓ Check GPU (RTX 4090)
- ✓ Download model (~128MB)
- ✓ Create directories
- ✓ Verify setup

---

## 🎬 DURING PRESENTATION

### Step 1: Introduction (30 seconds)
**Say**: "I'm demonstrating OpenAI's Guided Diffusion from 2021. It beat GANs with FID 3.94 vs StyleGAN's 7.49."

### Step 2: Show Installation (10 seconds)
```bash
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### Step 3: Generate WITHOUT Classifier (Baseline) (30 seconds)
```bash
./without_classifier.sh
```
**Say**: "First, baseline generation without classifier guidance. Takes ~8 seconds on RTX 4090!"

### Step 4: Generate WITH Classifier Guidance (60 seconds)
```bash
./with_classifier_guidance.sh
```
**Say**: "Now with classifier guidance - using gradients from a noisy classifier to steer generation toward more realistic images. Takes ~16 seconds due to gradient computation."

### Step 5: Compare Results (30 seconds)
```bash
./compare_results.sh
```
**Say**: "Let's compare them side-by-side. Notice how classifier guidance produces more realistic, class-specific features."

### Step 6: Show Comparison (30 seconds)
```bash
# View the comparison grid
xdg-open outputs/comparison/side_by_side.png
# Or just show: ls outputs/
```

---

## 🎯 CLASSIFIER GUIDANCE DEMO (Advanced)

### Quick Start - Complete Workflow
```bash
# 1. Generate baseline (without classifier)
./without_classifier.sh

# 2. Generate with classifier guidance  
./with_classifier_guidance.sh

# 3. Compare results side-by-side
./compare_results.sh

# 4. View comparison
ls outputs/
# - outputs/without_classifier/
# - outputs/with_classifier/
# - outputs/comparison/side_by_side.png
```

### Manual Testing (if needed)

#### Download Classifier Model
```bash
cd models
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt
cd ..
```

#### Test WITHOUT Classifier (Baseline)
```bash
python simple_demo.py \
    --attention_resolutions 32,16,8 --class_cond True \
    --diffusion_steps 1000 --dropout 0.1 --image_size 64 \
    --learn_sigma True --noise_schedule cosine \
    --num_channels 192 --num_head_channels 64 --num_res_blocks 3 \
    --resblock_updown True --use_new_attention_order True \
    --use_fp16 True --use_scale_shift_norm True \
    --model_path models/64x64_diffusion.pt \
    --timestep_respacing 250 --num_samples 4 --batch_size 2 \
    --output_dir outputs/without_classifier
```

#### Test WITH Classifier Guidance
```bash
python simple_demo.py \
    --attention_resolutions 32,16,8 --class_cond True \
    --diffusion_steps 1000 --dropout 0.1 --image_size 64 \
    --learn_sigma True --noise_schedule cosine \
    --num_channels 192 --num_head_channels 64 --num_res_blocks 3 \
    --resblock_updown True --use_new_attention_order True \
    --use_fp16 True --use_scale_shift_norm True \
    --model_path models/64x64_diffusion.pt \
    --classifier_path models/64x64_classifier.pt \
    --classifier_scale 1.0 --classifier_depth 4 \
    --timestep_respacing 250 --num_samples 4 --batch_size 2 \
    --output_dir outputs/with_classifier
```

### Compare Classifier Scales
Try different guidance scales to see the quality/diversity trade-off:
- `--classifier_scale 0.0` = No guidance (same as unconditional)
- `--classifier_scale 0.5` = Light guidance
- `--classifier_scale 1.0` = Normal guidance (recommended)
- `--classifier_scale 2.0` = Strong guidance (higher quality, less diversity)
- `--classifier_scale 5.0` = Very strong (may oversaturate)

**Key Point**: Higher scale = better quality but less diversity

---

## 💬 KEY TALKING POINTS

1. **What it does**: Generates images from noise using diffusion
2. **How**: Reverses noise addition process (1000 steps)
3. **Innovation**: Classifier guidance beats GANs
4. **Architecture**: U-Net with attention (280M params)
5. **Results**: FID 3.94 (ImageNet 256×256)

---

## ❓ EXPECTED QUESTIONS

**Q: Training time?**
A: "~7 days on 8 V100s. Using pre-trained weights here."

**Q: vs Stable Diffusion?**
A: "This (2021) works in pixel space. Stable Diffusion (2022) uses latent space. This came first."

**Q: How fast?**
A: "250 steps = only 6 seconds per batch on RTX 4090! Much faster than expected."

**Q: FID score?**
A: "3.94 on ImageNet 256×256. Beat StyleGAN2 (7.49) and BigGAN (6.95)."

---

## 🚨 IF THINGS GO WRONG

### Model download fails:
```bash
# Just show the code
cat guided_diffusion/gaussian_diffusion.py | head -100
# Explain the math instead
```

### Generation too slow:
```bash
Ctrl+C  # Cancel it
# Show pre-prepared figures from README
```

### No time:
```bash
# Just show PROJECT_OVERVIEW.md
# Walk through the architecture diagram
```

---

## 📊 KEY NUMBERS TO MEMORIZE

- **FID**: 3.94 (lower = better)
- **Parameters**: 280 million
- **Steps**: 1000 (or 250 fast)
- **Time**: ~30-60 sec for 4 images @ 64×64
- **GPU**: RTX 4090
- **Published**: May 2021

---

## ✅ SUCCESS CRITERIA

Minimum to show:
- ✓ Installation works
- ✓ GPU detected
- ✓ Code runs
- ✓ At least 1 image generated

Bonus points:
- ✓ Live generation demo
- ✓ Multiple samples
- ✓ Clear explanation
- ✓ Answer questions

---

## 🎯 THE PERFECT 30-SECOND PITCH

"This is OpenAI's Guided Diffusion from 2021. It generates images by reversing a noise process - starting from random noise and iteratively denoising over 250 steps. The key innovation is classifier guidance: using gradients from a classifier trained on noisy images to steer generation. This achieved FID 3.94, beating all GANs at the time. I have it running on an RTX 4090, and as you can see [show output], it generates realistic images in about a minute."

---

**YOU'VE GOT THIS! 🚀**
