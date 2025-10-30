# 🎓 Guided Diffusion - Live Demonstration Guide

**Presenter**: Senum  
**Hardware**: NVIDIA RTX 4090  
**Date**: October 30, 2025

---

## ⏱️ **5-Minute Quick Demo** (If Time is Short)

### Option 1: FASTEST - Show Pre-Generated Results
```bash
# Just explain the architecture and show the paper results
# Point to the README.md statistics
```

### Option 2: FAST - Download & Show Small Model
```bash
# Download smallest model (128MB, ~2 min download)
cd /home/senum/projects/guided-diffusion/guided-diffusion
wget -q --show-progress https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_diffusion.pt -P models/

# Generate 4 images in ~30 seconds
python scripts/image_sample.py \
    --model_path models/64x64_diffusion.pt \
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
    --use_scale_shift_norm True \
    --timestep_respacing 250 \
    --num_samples 4 \
    --batch_size 4

# Convert and display
python demo_quick_view.py
```

---

## 🎯 **15-Minute Full Demo** (Recommended)

### **Phase 1: Introduction (2 minutes)**

**What to Say:**
```
"I'm demonstrating OpenAI's Guided Diffusion model, published in 2021.
This paper showed that diffusion models can beat GANs in image quality.

Key Innovation: Classifier Guidance
- Train a classifier on noisy images
- Use gradients to guide generation
- Result: Better quality than StyleGAN2

I have it running on an RTX 4090 GPU."
```

**Show Files:**
- `README.md` - Official documentation
- `PROJECT_OVERVIEW.md` - Technical details
- `setup.py` - Installation proof

### **Phase 2: Architecture Explanation (3 minutes)**

**What to Say:**
```
"The system has three main components:

1. DIFFUSION MODEL (unet.py)
   - U-Net architecture with attention
   - Predicts noise at each timestep
   - 280M parameters

2. GAUSSIAN DIFFUSION (gaussian_diffusion.py)
   - Forward: Add noise gradually (1000 steps)
   - Reverse: Denoise to generate images
   - Supports DDIM for faster sampling

3. CLASSIFIER GUIDANCE (classifier_sample.py)
   - Classifier trained on noisy ImageNet
   - Provides gradients: ∇_x log p(y|x_t)
   - Steers generation toward target class"
```

**Show Code:**
```bash
# Show the U-Net model
cat guided_diffusion/unet.py | head -50

# Show the diffusion process
cat guided_diffusion/gaussian_diffusion.py | head -100
```

### **Phase 3: Live Generation (8 minutes)**

#### **Step 1: Download Model** (2 minutes)
```bash
cd /home/senum/projects/guided-diffusion/guided-diffusion

# Download 64x64 model (fastest for demo)
echo "Downloading 64x64 diffusion model..."
wget -q --show-progress \
    https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_diffusion.pt \
    -P models/

echo "✓ Model downloaded (128MB)"
ls -lh models/
```

#### **Step 2: Generate Images** (4 minutes)
```bash
# Create a quick sampling script
cat > quick_demo.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting image generation..."
echo "Configuration: 64x64 resolution, 4 samples, 250 steps"
echo ""

python scripts/image_sample.py \
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
    --batch_size 4 \
    2>&1 | grep -E "(creating|sampling|created|saving)"

echo ""
echo "✓ Generation complete!"
EOF

chmod +x quick_demo.sh
./quick_demo.sh
```

**What to Say While Running:**
```
"The model is now:
1. Loading the 280M parameter U-Net
2. Starting from pure Gaussian noise
3. Iteratively denoising over 250 steps
4. Each step predicts and removes noise
5. With RTX 4090, this takes about 30-60 seconds

The timestep_respacing=250 means we're using 250 steps
instead of 1000, which is 4x faster with minimal quality loss."
```

#### **Step 3: Display Results** (2 minutes)
```bash
# Create visualization script
cat > demo_quick_view.py << 'EOF'
import numpy as np
from PIL import Image
import os
import glob

# Find the generated samples
npz_files = glob.glob('samples_*.npz')
if not npz_files:
    print("❌ No samples found!")
    exit(1)

latest_file = max(npz_files, key=os.path.getctime)
print(f"📂 Loading: {latest_file}")

# Load samples
data = np.load(latest_file)
images = data['arr_0']
print(f"✓ Loaded {len(images)} images of shape {images[0].shape}")

# Save individual images
os.makedirs('demo_output', exist_ok=True)
for i, img in enumerate(images):
    save_path = f'demo_output/sample_{i+1}.png'
    Image.fromarray(img).save(save_path)
    print(f"✓ Saved: {save_path}")

# Create grid
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    if i < len(images):
        ax.imshow(images[i])
        ax.set_title(f'Sample {i+1}')
        ax.axis('off')
plt.tight_layout()
plt.savefig('demo_output/grid.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Grid saved: demo_output/grid.png")
print(f"\n🎉 Open demo_output/ folder to view all images!")
EOF

python demo_quick_view.py

# Show the images
ls -lh demo_output/
```

### **Phase 4: Q&A Preparation (2 minutes)**

---

## 💡 **Expected Questions & Answers**

### Q1: "How long does training take?"
**A**: "The pre-trained model took ~7 days on 8 V100 GPUs for 256×256 ImageNet. 
That's about 1 million iterations. For demo purposes, we're using pre-trained weights."

### Q2: "What's the difference from Stable Diffusion?"
**A**: "Great question! This (2021) works in pixel space and came first. 
Stable Diffusion (2022) works in latent space using an autoencoder, 
which is more memory efficient. This laid the groundwork for Stable Diffusion."

### Q3: "Can you control what images it generates?"
**A**: "Yes, two ways:
1. Class-conditional: Specify ImageNet classes (dogs, cars, etc.)
2. Classifier guidance: Use gradients to steer toward target classes
For text prompts, you'd need to integrate CLIP (not in this repo)."

### Q4: "What's the FID score?"
**A**: "For 256×256 ImageNet with guidance: FID of 3.94
This beat StyleGAN2's FID of 7.49 at the time.
Lower FID = better quality. This was state-of-the-art in 2021."

### Q5: "How does classifier guidance work?"
**A**: "The classifier is trained on noisy images at all timesteps.
During generation, we compute: gradient = ∇_x log p(y|x_t)
We add this gradient to the denoising step, pushing the sample
toward the target class. It's like gradient ascent on the class probability."

### Q6: "Why is sampling slow?"
**A**: "Standard sampling uses 1000 steps (the reverse diffusion process).
We can speed it up with:
- DDIM: Deterministic, can use 50-250 steps
- Timestep respacing: Skip steps strategically
- Better hardware: RTX 4090 helps a lot!
In my demo, 250 steps takes ~30-60 seconds per batch."

### Q7: "What datasets were used?"
**A**: "Three main datasets:
1. ImageNet (ILSVRC 2012): 1000 classes, ~1.3M images
2. LSUN Bedroom/Cat/Horse: Specific scene categories
3. All images resized to 64×64, 128×128, 256×256, or 512×512"

### Q8: "Can this run without GPU?"
**A**: "Technically yes, but it would be extremely slow.
On CPU, a single 256×256 image could take 10-20 minutes.
GPU is essential for practical use."

---

## 🎬 **Backup Plans**

### If Download Fails:
```bash
# Show the code architecture instead
tree -L 2 guided_diffusion/
cat guided_diffusion/gaussian_diffusion.py | head -200
```

### If Generation Takes Too Long:
```bash
# Kill it and show pre-made results
Ctrl+C
# Show paper figures from README.md
# Explain: "In production, we'd use cached results"
```

### If No Internet:
```bash
# Focus on code walkthrough
# Show the training pipeline
cat scripts/image_train.py
# Explain the architecture
# Show the mathematical formulas
```

---

## 📊 **Key Metrics to Memorize**

| Model | Resolution | FID | Time |
|-------|------------|-----|------|
| This work (guided) | 256×256 | **3.94** | ~1 min/image |
| StyleGAN2 | 256×256 | 7.49 | ~1 sec/image |
| BigGAN | 256×256 | 6.95 | ~1 sec/image |

**Key Point**: Better quality (lower FID) but slower sampling

---

## 🗣️ **Talking Points Summary**

1. **Innovation**: First to show diffusion beats GANs
2. **Method**: Classifier guidance with noisy images
3. **Architecture**: U-Net with attention, 280M params
4. **Results**: FID 3.94 on ImageNet 256×256
5. **Process**: 1000-step diffusion (or 250 with respacing)
6. **Hardware**: Runs on RTX 4090, generates 64×64 in ~30 sec

---

## ✅ **Pre-Demo Checklist**

```bash
# Run this before your presentation:
cd /home/senum/projects/guided-diffusion/guided-diffusion

# 1. Check installation
python -c "import guided_diffusion; print('✓ Package OK')"

# 2. Check GPU
python -c "import torch; print(f'✓ CUDA: {torch.cuda.is_available()}')"

# 3. Check disk space (need ~1GB for model)
df -h /home/senum

# 4. Test internet (for model download)
ping -c 2 openaipublic.blob.core.windows.net

# 5. Create directories
mkdir -p models demo_output

# 6. Pre-download model (optional but recommended!)
wget -q --show-progress \
    https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_diffusion.pt \
    -P models/

echo "✅ All systems ready for demo!"
```

---

## 🎯 **The Perfect 10-Minute Demo Flow**

```
[0:00-2:00]  Introduction + Paper Context
[2:00-4:00]  Code Architecture Walkthrough
[4:00-5:00]  Download Model
[5:00-9:00]  Generate Images (live)
[9:00-10:00] Show Results + Conclusion
```

---

## 🚨 **Emergency: If Everything Fails**

Show this **pre-prepared explanation**:

```bash
# Open PROJECT_OVERVIEW.md and explain:
# 1. The U-Net architecture diagram
# 2. The diffusion process equations
# 3. The published results table
# 4. The code structure

# Then say:
"Due to [download speed/time constraints/network issues], 
I've prepared a detailed technical overview instead. 
The key contribution is classifier guidance, which you can 
see implemented in classifier_sample.py..."
```

---

## 📝 **Script to Read Verbatim** (Emergency Backup)

```
"Good [morning/afternoon]. I'm demonstrating OpenAI's Guided Diffusion,
a 2021 breakthrough that proved diffusion models can surpass GANs.

The key innovation is classifier guidance. We train a classifier on noisy 
images, then use its gradients to steer generation toward realistic outputs.

The architecture uses a U-Net with attention mechanisms. It has 280 million 
parameters and achieves an FID score of 3.94 on ImageNet 256x256, beating 
StyleGAN2's 7.49.

I've installed it successfully on an RTX 4090. The generation process 
starts with pure noise and iteratively denoises over 1000 steps, or 250 
with DDIM acceleration.

[If live demo works]: As you can see, it generates high-quality images...
[If no demo]: The results published show photorealistic quality...

This work laid the foundation for modern diffusion models like Stable 
Diffusion and DALL-E 2. Thank you."
```

---

**Good luck! 🍀 You've got this! 🚀**
