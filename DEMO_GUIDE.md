# 🎓 Live Demonstration Guide for Guided Diffusion

## 📋 Pre-Demo Checklist (DO THIS BEFORE THE PRESENTATION)

### ✅ Step 1: Environment Setup (5 minutes)

```bash
# Navigate to project directory
cd /home/senum/projects/guided-diffusion/guided-diffusion

# Install the package
pip install -e .

# Install missing dependencies
pip install torch torchvision torchaudio
pip install blobfile>=1.0.5

# Verify installation
python -c "import guided_diffusion; print('✓ Package installed successfully')"
```

### ✅ Step 2: Create Required Directories

```bash
# Create directories for models and outputs
mkdir -p models
mkdir -p outputs
mkdir -p logs
```

### ✅ Step 3: Download Pre-trained Models (CHOOSE ONE OPTION)

**OPTION A: Small & Fast Demo (Recommended for Live Demo)**
```bash
# Download 64x64 models (faster, smaller files)
cd models
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_diffusion.pt
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt
cd ..
```

**OPTION B: High Quality Demo (If you have time/bandwidth)**
```bash
# Download 256x256 models (better quality but slower)
cd models
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt
cd ..
```

**OPTION C: LSUN Bedroom (Fastest, No Classifier Needed)**
```bash
# Download LSUN bedroom model (unconditional, very impressive)
cd models
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_bedroom.pt
cd ..
```

### ✅ Step 4: Test Run (Before the Demo!)

```bash
# Quick test to ensure everything works
python scripts/image_sample.py \
    --attention_resolutions 32,16,8 \
    --class_cond False \
    --diffusion_steps 1000 \
    --image_size 64 \
    --learn_sigma True \
    --noise_schedule linear \
    --num_channels 192 \
    --num_head_channels 64 \
    --num_res_blocks 3 \
    --resblock_updown True \
    --use_fp16 False \
    --use_scale_shift_norm True \
    --timestep_respacing 50 \
    --model_path models/64x64_diffusion.pt \
    --batch_size 2 \
    --num_samples 2

# This should generate: samples_2x64x64x3.npz
```

---

## 🎬 LIVE DEMONSTRATION SCRIPT (During Presentation)

### Part 1: Introduction (2 minutes)

**What to Say:**
```
"Today I'll demonstrate Guided Diffusion, OpenAI's breakthrough model 
that proved diffusion models can beat GANs for image generation. This 
was published in May 2021 and achieved state-of-the-art results on 
ImageNet with FID scores as low as 2.07 for 64x64 images."
```

**Show:**
1. Open `README.md` - briefly show the paper reference
2. Open `PROJECT_OVERVIEW.md` - show the architecture diagram

---

### Part 2: Code Walkthrough (3-4 minutes)

**Demo 1: Show the Diffusion Process**

```bash
# Open the main diffusion file
code guided_diffusion/gaussian_diffusion.py
```

**What to Explain:**
```
"The core of diffusion models is here. The forward process (q_sample) 
adds noise gradually, and the reverse process (p_sample) removes it 
step by step. This is fundamentally different from GANs which generate 
in one shot."
```

**Point out these key functions:**
- Line ~190: `q_sample()` - forward diffusion (adding noise)
- Line ~245: `p_sample()` - reverse diffusion (denoising)
- Line ~580: `training_losses()` - how the model learns

**Demo 2: Show the U-Net Architecture**

```bash
# Open the U-Net model
code guided_diffusion/unet.py
```

**What to Explain:**
```
"The U-Net architecture predicts noise at each timestep. It has an 
encoder-decoder structure with skip connections and attention mechanisms 
at multiple resolutions."
```

**Point out:**
- Line ~350: `UNetModel` class definition
- Line ~400-450: Encoder blocks with downsampling
- Line ~500-550: Decoder blocks with upsampling
- Line ~580: Forward pass combining everything

---

### Part 3: Live Generation (5-7 minutes)

**Demo 3a: Generate Images WITHOUT Classifier Guidance (Faster)**

```bash
# Option 1: LSUN Bedroom (Most Impressive, Fastest)
python scripts/image_sample.py \
    --attention_resolutions 32,16,8 \
    --class_cond False \
    --diffusion_steps 1000 \
    --dropout 0.1 \
    --image_size 256 \
    --learn_sigma True \
    --noise_schedule linear \
    --num_channels 256 \
    --num_head_channels 64 \
    --num_res_blocks 2 \
    --resblock_updown True \
    --use_fp16 False \
    --use_scale_shift_norm True \
    --timestep_respacing 100 \
    --model_path models/lsun_bedroom.pt \
    --batch_size 4 \
    --num_samples 4
```

**What to Say While Running:**
```
"I'm generating 4 bedroom images using 100 denoising steps. The model 
was trained on LSUN bedroom dataset. Notice it's an iterative process - 
the model starts from pure noise and gradually removes it to create 
photorealistic bedroom scenes."

"The timestep_respacing parameter lets us use 100 steps instead of 1000, 
making it 10x faster with minimal quality loss. This is called DDIM 
sampling."
```

**Demo 3b: Generate Images WITH Classifier Guidance (If time permits)**

```bash
# ImageNet with classifier guidance
python scripts/classifier_sample.py \
    --attention_resolutions 32,16,8 \
    --class_cond True \
    --diffusion_steps 1000 \
    --image_size 64 \
    --learn_sigma True \
    --noise_schedule cosine \
    --num_channels 192 \
    --num_head_channels 64 \
    --num_res_blocks 3 \
    --resblock_updown True \
    --use_fp16 False \
    --use_scale_shift_norm True \
    --use_new_attention_order True \
    --timestep_respacing 100 \
    --model_path models/64x64_diffusion.pt \
    --classifier_path models/64x64_classifier.pt \
    --classifier_scale 1.0 \
    --batch_size 4 \
    --num_samples 4
```

**What to Say:**
```
"Now with classifier guidance. The classifier was trained on noisy images 
at all noise levels. During generation, it computes gradients that steer 
the process toward more realistic, class-specific images. The classifier_scale 
parameter controls this - higher values give better quality but less diversity."
```

---

### Part 4: Visualize Results (2-3 minutes)

**Create Visualization Script:**

```bash
# Create a quick visualization script
cat > visualize_samples.py << 'EOF'
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os

# Find the most recent samples file
import glob
sample_files = sorted(glob.glob('samples_*.npz'))
if not sample_files:
    print("No sample files found!")
    sys.exit(1)

latest_file = sample_files[-1]
print(f"Loading: {latest_file}")

# Load samples
data = np.load(latest_file)
images = data['arr_0']
print(f"Shape: {images.shape}")

# Create output directory
os.makedirs('outputs', exist_ok=True)

# Save individual images
for i, img in enumerate(images):
    Image.fromarray(img).save(f'outputs/sample_{i:04d}.png')
    print(f"Saved: outputs/sample_{i:04d}.png")

# Create grid visualization
n_images = min(len(images), 16)
n_cols = 4
n_rows = (n_images + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes

for i in range(len(axes)):
    if i < n_images:
        axes[i].imshow(images[i])
        axes[i].set_title(f'Sample {i+1}')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('outputs/grid.png', dpi=150, bbox_inches='tight')
print("Saved: outputs/grid.png")
plt.show()

print(f"\n✓ Generated {len(images)} images successfully!")
EOF

# Run visualization
python visualize_samples.py
```

**What to Show:**
1. Open the `outputs/grid.png` file to show all generated images
2. Open individual images to show detail
3. Zoom in to show the quality

**What to Say:**
```
"As you can see, the generated images are photorealistic. The model 
learned the data distribution through the diffusion process, not through 
adversarial training like GANs. This makes training more stable and 
results in better mode coverage."
```

---

### Part 5: Technical Highlights (2 minutes)

**Open terminal and explain the architecture:**

```bash
# Show model size
ls -lh models/*.pt

# Show the training utilities
code guided_diffusion/train_util.py
```

**Key Points to Mention:**
```
✓ "The model uses a U-Net with attention at multiple resolutions"
✓ "Training is stable - no mode collapse like GANs"
✓ "The classifier guidance was the key innovation here"
✓ "Achieved FID scores of 2.07 on ImageNet 64x64, beating BigGAN"
✓ "Supports distributed training across multiple GPUs"
✓ "Can use DDIM for 10-50x faster sampling"
```

---

### Part 6: Results & Comparison (1-2 minutes)

**Show the results table from PROJECT_OVERVIEW.md:**

```bash
code PROJECT_OVERVIEW.md
# Scroll to "Results" section
```

**What to Say:**
```
"Let me show you the quantitative results:
- ImageNet 256x256: FID of 4.59 (better than BigGAN's 6.95)
- LSUN Bedroom: FID of 1.90 (state-of-the-art)
- Higher precision than GANs, meaning more realistic images
- Better recall than GANs, meaning better mode coverage

This paper was a turning point that showed diffusion models could 
beat GANs, leading to later developments like DALL-E 2, Imagen, 
and Stable Diffusion."
```

---

## 🎯 Quick Demo Scripts (If Short on Time)

### FASTEST DEMO (2-3 minutes total)

```bash
# Pre-download LSUN bedroom model before demo

# During demo:
python scripts/image_sample.py \
    --attention_resolutions 32,16,8 \
    --class_cond False \
    --diffusion_steps 1000 \
    --dropout 0.1 \
    --image_size 256 \
    --learn_sigma True \
    --noise_schedule linear \
    --num_channels 256 \
    --num_head_channels 64 \
    --num_res_blocks 2 \
    --resblock_updown True \
    --use_fp16 False \
    --use_scale_shift_norm True \
    --timestep_respacing 50 \
    --model_path models/lsun_bedroom.pt \
    --batch_size 2 \
    --num_samples 2

# Visualize immediately
python visualize_samples.py
```

---

## 🔧 Troubleshooting During Demo

### If Generation is Too Slow:
```bash
# Reduce timesteps
--timestep_respacing 25  # Instead of 100 or 250

# Reduce batch size
--batch_size 1

# Reduce number of samples
--num_samples 1
```

### If Out of Memory:
```bash
# Use smaller model
--use_fp16 False  # Might help

# Reduce batch size
--batch_size 1

# Use smaller image size (if you have 64x64 model)
--image_size 64
```

### If Model Download Fails:
```bash
# Alternative: Use curl
curl -O https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_bedroom.pt

# Or use aria2c (faster)
aria2c -x 16 https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_bedroom.pt
```

---

## 🎤 Key Talking Points Summary

### Technical Achievements:
1. **Beat GANs**: First diffusion model to surpass GAN quality
2. **Classifier Guidance**: Novel technique using gradients from noisy classifier
3. **Stable Training**: No adversarial training, no mode collapse
4. **DDIM**: Fast sampling technique (25-250 steps vs 1000)

### Architecture Highlights:
1. **U-Net with Attention**: Multi-scale architecture
2. **Timestep Conditioning**: Model knows which noise level to denoise
3. **Learned Variance**: Model can learn optimal noise schedule
4. **Scale-Shift Normalization**: Better conditioning mechanism

### Practical Advantages:
1. **Better Mode Coverage**: Captures more diversity in data
2. **Stable Training**: Doesn't require careful GAN balancing
3. **Controllable Trade-off**: Quality vs diversity via guidance scale
4. **Extensible**: Led to DALL-E 2, Imagen, Stable Diffusion

---

## 📊 Expected Demo Timeline

| Time | Activity | Duration |
|------|----------|----------|
| 0:00-2:00 | Introduction & Background | 2 min |
| 2:00-5:00 | Code Walkthrough | 3 min |
| 5:00-10:00 | Live Generation | 5 min |
| 10:00-12:00 | Show Results | 2 min |
| 12:00-14:00 | Technical Discussion | 2 min |
| 14:00-15:00 | Q&A | 1 min |

**Total: ~15 minutes**

---

## ✅ Final Checklist Before Demo

- [ ] All dependencies installed (`pip install -e .`)
- [ ] At least one model downloaded (lsun_bedroom.pt recommended)
- [ ] Test run completed successfully
- [ ] Visualization script created and tested
- [ ] `PROJECT_OVERVIEW.md` open in editor for reference
- [ ] Terminal ready with commands prepared
- [ ] Output directory exists and is empty
- [ ] Backup plan: Have pre-generated samples ready just in case

---

## 💡 Pro Tips

1. **Practice the demo at least once before presenting**
2. **Have pre-generated samples as backup** in case live generation fails
3. **Keep the visualization script open** for quick results display
4. **Use LSUN bedroom model** - it's fastest and most impressive
5. **Prepare for questions** about:
   - How diffusion differs from GANs
   - Why it's better than GANs
   - Computational cost
   - Applications (DALL-E 2, Stable Diffusion)

---

## 🎓 Likely Questions & Answers

**Q: How long does training take?**
A: "On 8 V100 GPUs, the 256x256 ImageNet model takes about 7 days. But we're using pre-trained models today."

**Q: Is this faster than GANs?**
A: "Generation is slower (100-1000 steps vs 1 forward pass), but training is more stable and quality is better. DDIM helps speed up sampling."

**Q: What's the advantage over GANs?**
A: "No mode collapse, better mode coverage, stable training, and ultimately better image quality as measured by FID scores."

**Q: Can I use this for my own data?**
A: "Yes! The code supports custom datasets. You'd need to train from scratch or fine-tune a pre-trained model."

**Q: How does this relate to Stable Diffusion?**
A: "This was a foundational work. Stable Diffusion builds on these ideas but works in latent space for efficiency, and adds text conditioning via CLIP."

---

Good luck with your demonstration! 🚀
