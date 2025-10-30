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

### Step 3: Start Generation (30 seconds)
```bash
./quick_demo.sh
```
**Say**: "Starting diffusion process - 250 steps of iterative denoising. Takes only 6 seconds on RTX 4090!"

### Step 4: While Running, Explain (during the ~60 seconds)
**Say**: 
- "U-Net architecture with 280M parameters"
- "Starts from random noise, removes it step-by-step"
- "Uses classifier gradients for guidance"
- "Process: x_T → x_{T-1} → ... → x_0"

### Step 5: Show Results (30 seconds)
```bash
ls demo_output/
# Images are automatically saved and displayed
```

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
