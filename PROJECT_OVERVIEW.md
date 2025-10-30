# Guided Diffusion - Comprehensive Project Documentation

## Table of Contents
- [Overview](#overview)
- [What This Project Does](#what-this-project-does)
- [Key Concepts](#key-concepts)
- [Architecture](#architecture)
- [Core Components](#core-components)
- [Workflow](#workflow)
- [Pre-trained Models](#pre-trained-models)
- [Usage Examples](#usage-examples)
- [Technical Details](#technical-details)
- [Results](#results)
- [Installation](#installation)

---

## Overview

This is the official PyTorch implementation of **"Diffusion Models Beat GANs on Image Synthesis"** ([arXiv:2105.05233](http://arxiv.org/abs/2105.05233)), a groundbreaking paper by OpenAI researchers published in 2021. This work demonstrated that diffusion models could surpass GANs in image generation quality, marking a paradigm shift in generative AI.

**Paper Authors**: Prafulla Dhariwal and Alex Nichol  
**Published**: May 2021  
**Based on**: [openai/improved-diffusion](https://github.com/openai/improved-diffusion)

---

## What This Project Does

### Primary Capabilities

1. **High-Quality Image Generation**: Generate photorealistic images at multiple resolutions (64×64 to 512×512)
2. **Class-Conditional Generation**: Create images of specific categories (e.g., "golden retriever", "volcano")
3. **Classifier Guidance**: Steer generation toward more realistic outputs using gradient-based guidance
4. **Image Upsampling**: Super-resolution from low to high resolution (64→256, 128→512)
5. **Training Infrastructure**: Complete pipeline for training your own diffusion models and classifiers

### Supported Datasets

- **ImageNet (ILSVRC 2012)**: 1000 classes, ~1M images
- **LSUN**: Specific categories (bedroom, cat, horse)
- **Custom datasets**: Extensible to your own data

---

## Key Concepts

### 1. Diffusion Models

Diffusion models generate images through a two-stage process:

**Forward Process (Noise Addition)**:
```
x₀ → x₁ → x₂ → ... → x_T
Clean Image → Gradually Noisier → Pure Noise
```

**Reverse Process (Denoising)**:
```
x_T → x_{T-1} → ... → x₁ → x₀
Pure Noise → Gradually Cleaner → Generated Image
```

The model learns to predict and remove noise at each step, effectively learning the data distribution.

### 2. Classifier Guidance

The key innovation of this work: use a classifier trained on noisy images to guide generation.

**How it works**:
- Train a classifier that can recognize ImageNet classes even in noisy images
- During sampling, compute classifier gradients: ∇_x log p(y|x_t)
- Use these gradients to push the sample toward the target class
- Controlled by a guidance scale parameter (higher = more realistic but less diverse)

**Benefits**:
- Significantly improved image quality (lower FID scores)
- Better class-conditional generation
- Trade-off between sample quality and diversity

### 3. DDIM Sampling

**DDIM** (Denoising Diffusion Implicit Models) provides:
- Deterministic sampling (same seed → same image)
- Faster generation (25-250 steps vs 1000)
- Maintains quality while reducing computational cost

---

## Architecture

### U-Net Model Structure

```
Input Image (noisy x_t) + Timestep t + [Optional: Class Label y]
    ↓
┌─────────────────────────────────────────┐
│   Timestep Embedding (sinusoidal)       │
│   + Class Embedding (if conditional)    │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│          ENCODER (Downsampling)         │
│  ┌────────────────────────────────┐     │
│  │  ResBlock + Attention (32×32)  │     │
│  │  ↓ Downsample                  │     │
│  │  ResBlock + Attention (16×16)  │     │
│  │  ↓ Downsample                  │     │
│  │  ResBlock + Attention (8×8)    │     │
│  └────────────────────────────────┘     │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│         MIDDLE (Bottleneck)             │
│  ResBlock + Attention + ResBlock        │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│          DECODER (Upsampling)           │
│  ┌────────────────────────────────┐     │
│  │  ResBlock + Attention (8×8)    │     │
│  │  ↑ Upsample                    │     │
│  │  ResBlock + Attention (16×16)  │     │
│  │  ↑ Upsample                    │     │
│  │  ResBlock + Attention (32×32)  │     │
│  └────────────────────────────────┘     │
│  (Skip connections from encoder)        │
└─────────────────────────────────────────┘
    ↓
Output: Predicted Noise ε_θ(x_t, t) [or x₀ or x_{t-1}]
```

### Key Architectural Features

1. **ResBlocks with Scale-Shift Normalization**: Better conditioning on timesteps
2. **Multi-Head Self-Attention**: Captures long-range dependencies at multiple resolutions
3. **Adaptive Group Normalization**: Improved stability
4. **Skip Connections**: Preserve fine details from encoder to decoder
5. **Learned Variance**: Model can learn optimal noise schedule

---

## Core Components

### 1. `gaussian_diffusion.py` - Diffusion Process

**Key Classes**:
- `GaussianDiffusion`: Main diffusion process implementation

**Important Methods**:

```python
# Forward diffusion: Add noise to clean images
q_sample(x_start, t, noise)
# Returns: Noisy version of x_start at timestep t

# Reverse diffusion: Predict and remove noise
p_sample(model, x, t, clip_denoised, cond_fn, model_kwargs)
# Returns: x_{t-1} (one step cleaner)

# Training: Compute loss
training_losses(model, x_start, t, model_kwargs, noise)
# Returns: Loss dict with MSE and optional VB terms

# DDIM sampling (faster)
ddim_sample(model, x, t, eta=0.0)
# Returns: Deterministic denoised sample
```

**Noise Schedules**:
- **Linear**: β increases linearly (standard)
- **Cosine**: Smoother noise schedule, better for high-res images

### 2. `unet.py` - Neural Network Architecture

**Model Variants**:

```python
# Standard diffusion model
UNetModel(image_size, in_channels, model_channels, out_channels, ...)

# Super-resolution model (conditioned on low-res image)
SuperResModel(image_size, in_channels, ...)

# Classifier for guidance (encoder-only)
EncoderUNetModel(image_size, in_channels, ..., pool='attention')
```

**Key Parameters**:
- `image_size`: Target resolution (64, 128, 256, 512)
- `model_channels`: Base channel count (controls model size)
- `num_res_blocks`: Residual blocks per resolution level
- `attention_resolutions`: Where to apply attention (e.g., "32,16,8")
- `channel_mult`: Channel multipliers per level (e.g., (1,2,4,8))
- `num_head_channels`: Attention head dimension
- `use_scale_shift_norm`: Use FiLM-like conditioning
- `dropout`: Dropout rate for regularization

### 3. Training Scripts

#### `image_train.py` - Train Diffusion Models

```bash
python scripts/image_train.py \
    --data_dir /path/to/imagenet \
    --image_size 256 \
    --num_channels 256 \
    --num_res_blocks 2 \
    --learn_sigma True \
    --class_cond True \
    --batch_size 4 \
    --lr 1e-4 \
    --save_interval 10000
```

#### `classifier_train.py` - Train Noisy Classifiers

```bash
mpiexec -n 8 python scripts/classifier_train.py \
    --data_dir /path/to/imagenet \
    --image_size 256 \
    --classifier_attention_resolutions 32,16,8 \
    --classifier_depth 2 \
    --classifier_width 128 \
    --batch_size 256 \
    --iterations 300000
```

### 4. Sampling Scripts

#### `image_sample.py` - Unconditional/Conditional Sampling

```bash
python scripts/image_sample.py \
    --model_path models/256x256_diffusion.pt \
    --image_size 256 \
    --num_samples 100 \
    --batch_size 4 \
    --timestep_respacing 250
```

#### `classifier_sample.py` - Guided Sampling

```bash
python scripts/classifier_sample.py \
    --model_path models/256x256_diffusion.pt \
    --classifier_path models/256x256_classifier.pt \
    --classifier_scale 1.0 \
    --image_size 256 \
    --num_samples 100
```

### 5. `script_util.py` - Configuration & Model Creation

Helper functions for creating models with proper hyperparameters:
- `model_and_diffusion_defaults()`: Default training config
- `create_model_and_diffusion()`: Instantiate model and diffusion
- `create_classifier()`: Instantiate classifier for guidance

### 6. `train_util.py` - Training Loop

**TrainLoop Class**: Complete training infrastructure
- Mixed precision training (FP16)
- Distributed training (DDP)
- EMA (Exponential Moving Average) of parameters
- Gradient accumulation (microbatching)
- Learning rate annealing
- Checkpoint saving/loading

### 7. `respace.py` - Timestep Respacing

**SpacedDiffusion**: Skip timesteps for faster sampling
- Maintains quality while using fewer steps
- Example: Use 250 steps instead of 1000 (4× speedup)
- DDIM scheduling for optimal step selection

### 8. `evaluator.py` - Evaluation Metrics

Calculate standard generative model metrics:
- **FID** (Fréchet Inception Distance): Overall quality
- **Precision**: Fraction of generated samples that are realistic
- **Recall**: Fraction of real data modes covered
- Uses pre-trained Inception-V3 features

---

## Workflow

### Complete Training Pipeline

```
1. Data Preparation
   ├─ Download ImageNet or LSUN dataset
   ├─ Organize into directory structure
   └─ Preprocess to desired resolution

2. Train Diffusion Model
   ├─ Configure model architecture
   ├─ Set training hyperparameters
   ├─ Run distributed training (multiple GPUs)
   └─ Save checkpoints periodically

3. [Optional] Train Classifier
   ├─ Use same dataset
   ├─ Train on noisy images at all timesteps
   └─ Save classifier checkpoint

4. Generate Samples
   ├─ Load trained model checkpoint
   ├─ [Optional] Load classifier for guidance
   ├─ Sample with desired settings
   └─ Save generated images

5. Evaluate Quality
   ├─ Generate reference batch activations
   ├─ Generate sample batch activations
   ├─ Compute FID, Precision, Recall
   └─ Compare with baselines
```

### Sampling Process (Step-by-step)

```python
# Pseudocode for guided sampling

1. Initialize: x_T ~ N(0, I)  # Pure Gaussian noise
2. Select target class: y (e.g., y=207 for "golden retriever")

3. For t = T down to 1:
    # Predict noise using diffusion model
    ε_θ = model(x_t, t, y)
    
    # Get classifier gradients (guidance)
    if use_guidance:
        ∇_x log p(y|x_t) = classifier.grad(x_t, t, y)
    
    # Compute denoised prediction
    x_0_pred = (x_t - √(1-ᾱ_t) · ε_θ) / √ᾱ_t
    
    # Compute mean for x_{t-1}
    μ_t = ... # Using x_0_pred and x_t
    
    # Add guidance to mean
    if use_guidance:
        μ_t = μ_t + s · σ_t² · ∇_x log p(y|x_t)
    
    # Sample x_{t-1}
    x_{t-1} = μ_t + σ_t · z, where z ~ N(0,I)

4. Return: x_0 (final generated image)
```

---

## Pre-trained Models

### Available Checkpoints

All models available at: `https://openaipublic.blob.core.windows.net/diffusion/jul-2021/`

#### ImageNet Models (Class-Conditional)

| Resolution | Diffusion Model | Classifier | FID ↓ | Use Case |
|------------|----------------|------------|-------|----------|
| 64×64 | `64x64_diffusion.pt` | `64x64_classifier.pt` | 2.07 | Fast, high-quality |
| 128×128 | `128x128_diffusion.pt` | `128x128_classifier.pt` | 2.97 | Balanced |
| 256×256 | `256x256_diffusion.pt` | `256x256_classifier.pt` | 4.59 | High detail |
| 512×512 | `512x512_diffusion.pt` | `512x512_classifier.pt` | 7.72 | Maximum detail |

#### Unconditional Models

| Model | Checkpoint | Description |
|-------|------------|-------------|
| ImageNet 256×256 | `256x256_diffusion_uncond.pt` | No class conditioning |

#### Upsampling Models

| Model | Checkpoint | Purpose |
|-------|------------|---------|
| 64→256 | `64_256_upsampler.pt` | 4× super-resolution |
| 128→512 | `128_512_upsampler.pt` | 4× super-resolution |

#### LSUN Models (Class-Unconditional)

| Dataset | Checkpoint | FID ↓ |
|---------|------------|-------|
| LSUN Bedroom | `lsun_bedroom.pt` | 1.90 |
| LSUN Cat | `lsun_cat.pt` | 5.57 |
| LSUN Horse | `lsun_horse.pt` | 2.57 |
| LSUN Horse (no dropout) | `lsun_horse_nodropout.pt` | - |

---

## Usage Examples

### Example 1: Generate ImageNet Images with Guidance

```bash
# Download models
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt -P models/
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt -P models/

# Set model configuration
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 \
--image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 \
--num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True \
--use_scale_shift_norm True"

# Set sampling configuration
SAMPLE_FLAGS="--batch_size 4 --num_samples 100 --timestep_respacing 250"

# Generate samples with classifier guidance
python scripts/classifier_sample.py $MODEL_FLAGS \
    --model_path models/256x256_diffusion.pt \
    --classifier_path models/256x256_classifier.pt \
    --classifier_scale 1.0 \
    $SAMPLE_FLAGS
```

**Output**: `samples_100x256x256x3.npz` containing 100 generated images

### Example 2: Generate LSUN Bedroom Images

```bash
# Download model
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_bedroom.pt -P models/

# Set configuration
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 \
--dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear \
--num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True \
--use_fp16 True --use_scale_shift_norm True"

SAMPLE_FLAGS="--batch_size 4 --num_samples 50 --timestep_respacing 1000"

# Generate samples
python scripts/image_sample.py $MODEL_FLAGS \
    --model_path models/lsun_bedroom.pt \
    $SAMPLE_FLAGS
```

### Example 3: Fast Sampling with DDIM (25 steps)

```bash
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --image_size 128 \
--learn_sigma True --num_channels 256 --num_heads 4 --num_res_blocks 2 \
--resblock_updown True --use_fp16 True --use_scale_shift_norm True"

# Use DDIM with only 25 steps (40× faster than 1000 steps!)
SAMPLE_FLAGS="--batch_size 16 --num_samples 1000 --timestep_respacing ddim25 --use_ddim True"

python scripts/classifier_sample.py $MODEL_FLAGS \
    --model_path models/128x128_diffusion.pt \
    --classifier_path models/128x128_classifier.pt \
    --classifier_scale 1.0 \
    $SAMPLE_FLAGS
```

### Example 4: Image Super-Resolution

```bash
# First, generate low-resolution images
python scripts/image_sample.py \
    --model_path models/64x64_diffusion.pt \
    --image_size 64 \
    --num_samples 100 \
    --batch_size 16
# This creates 64_samples.npz

# Then, upsample to 256×256
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 \
--large_size 256 --small_size 64 --learn_sigma True --noise_schedule linear \
--num_channels 192 --num_heads 4 --num_res_blocks 2 --resblock_updown True \
--use_fp16 True --use_scale_shift_norm True"

python scripts/super_res_sample.py $MODEL_FLAGS \
    --model_path models/64_256_upsampler.pt \
    --base_samples 64_samples.npz \
    --batch_size 4 --num_samples 100 --timestep_respacing 250
```

### Example 5: Evaluate Generated Samples

```bash
# Generate reference batch (real images)
python evaluations/evaluator.py reference_batch.npz sample_batch.npz
```

This outputs:
- FID score
- Precision
- Recall
- Inception Score

---

## Technical Details

### Hyperparameters for Pre-trained Models

#### 256×256 ImageNet Model

```python
{
    'image_size': 256,
    'num_channels': 256,
    'num_res_blocks': 2,
    'num_heads': 4,
    'num_head_channels': 64,
    'attention_resolutions': '32,16,8',
    'channel_mult': '',  # Uses default: (1,1,2,2,4,4)
    'dropout': 0.0,
    'class_cond': True,
    'use_checkpoint': False,
    'use_scale_shift_norm': True,
    'resblock_updown': True,
    'use_fp16': True,
    'use_new_attention_order': False,
    'learn_sigma': True,
    'diffusion_steps': 1000,
    'noise_schedule': 'linear',
}
```

#### 256×256 Classifier

```python
{
    'image_size': 256,
    'classifier_width': 128,
    'classifier_depth': 2,
    'classifier_attention_resolutions': '32,16,8',
    'classifier_use_scale_shift_norm': True,
    'classifier_resblock_updown': True,
    'classifier_pool': 'attention',
}
```

### Guidance Scale Recommendations

| Scale | Effect | Use Case |
|-------|--------|----------|
| 0.0 | No guidance | Maximum diversity |
| 0.5 | Mild guidance | Balanced |
| 1.0 | Standard guidance | Default for 256×256 |
| 4.0 | Strong guidance | Maximum quality (512×512) |
| 10.0+ | Very strong | Unconditional → guided |

**Trade-off**: Higher scale → better quality but less diversity

### Training Time Estimates

(On 8× V100 GPUs)

| Model | Resolution | Training Time | Iterations |
|-------|------------|---------------|------------|
| Diffusion | 64×64 | ~3 days | ~500K |
| Diffusion | 256×256 | ~7 days | ~1M |
| Classifier | 128×128 | ~1 day | ~300K |

### Memory Requirements

| Resolution | Batch Size | GPU Memory | Model Size |
|------------|------------|------------|------------|
| 64×64 | 128 | ~16GB | ~280M params |
| 128×128 | 32 | ~16GB | ~280M params |
| 256×256 | 8 | ~16GB | ~280M params |
| 512×512 | 2 | ~16GB | ~280M params |

*Note: With gradient checkpointing enabled*

### Loss Functions

**MSE Loss** (default):
```
L_simple = E_t,x₀,ε [||ε - ε_θ(x_t, t)||²]
```

**VLB Loss** (variational lower bound):
```
L_vlb = E_t,x₀ [D_KL(q(x_{t-1}|x_t,x₀) || p_θ(x_{t-1}|x_t))]
```

**Hybrid Loss** (when learning variance):
```
L_hybrid = L_simple + λ · L_vlb
```

---

## Results

### Quantitative Results

#### ImageNet (Pure Guided Diffusion)

| Resolution | FID ↓ | Precision ↑ | Recall ↑ | IS ↑ |
|------------|-------|-------------|----------|------|
| 64×64 | **2.07** | 0.74 | 0.63 | - |
| 128×128 | **2.97** | 0.78 | 0.59 | - |
| 256×256 | **4.59** | 0.82 | 0.52 | - |
| 512×512 | **7.72** | 0.87 | 0.42 | - |

#### ImageNet (With Upsampling + Guidance)

| Resolution | FID ↓ | Precision ↑ | Recall ↑ |
|------------|-------|-------------|----------|
| 256×256 | **3.94** | 0.83 | 0.53 |
| 512×512 | **3.85** | 0.84 | 0.53 |

*Note: These results beat state-of-the-art GANs at the time*

#### LSUN (Unguided)

| Dataset | FID ↓ | Precision ↑ | Recall ↑ |
|---------|-------|-------------|----------|
| Bedroom | **1.90** | 0.66 | 0.51 |
| Cat | **5.57** | 0.63 | 0.52 |
| Horse | **2.57** | 0.71 | 0.55 |

### Comparison with GANs

At publication time (May 2021):

| Method | ImageNet 256×256 FID ↓ |
|--------|------------------------|
| **Guided Diffusion (this work)** | **3.94** |
| BigGAN-deep | 6.95 |
| StyleGAN2 + ADA | 7.49 |
| DDPM | 10.94 |

### Qualitative Observations

**Strengths**:
- ✅ Highly photorealistic images
- ✅ Better mode coverage than GANs (higher recall)
- ✅ No mode collapse issues
- ✅ Stable training (no adversarial dynamics)
- ✅ Controllable quality-diversity trade-off

**Limitations**:
- ⚠️ Slower sampling (1000 steps, though DDIM helps)
- ⚠️ Occasional unrealistic human faces
- ⚠️ Strong guidance reduces diversity
- ⚠️ Higher computational cost than GANs

---

## Installation

### Requirements

```bash
# Python 3.7+
pip install torch torchvision  # PyTorch 1.7+
pip install blobfile>=1.0.5
pip install tqdm
pip install mpi4py  # For distributed training
```

### Setup

```bash
# Clone repository
git clone https://github.com/openai/guided-diffusion.git
cd guided-diffusion

# Install package
pip install -e .

# Create directories
mkdir -p models logs data
```

### Verify Installation

```bash
python -c "import guided_diffusion; print('Success!')"
```

### Docker (Alternative)

```dockerfile
FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

WORKDIR /workspace
COPY . .
RUN pip install -e .
```

---

## Advanced Topics

### Custom Datasets

To train on your own dataset:

1. **Organize images**:
```
data/
  my_dataset/
    train/
      class1/
        img1.jpg
        img2.jpg
      class2/
        ...
```

2. **Modify `image_datasets.py`**:
```python
def load_data(data_dir, batch_size, image_size, class_cond=False):
    # Add your custom dataset loader
    ...
```

3. **Train**:
```bash
python scripts/image_train.py --data_dir data/my_dataset ...
```

### Fine-tuning

To fine-tune a pre-trained model:

```bash
python scripts/image_train.py \
    --resume_checkpoint models/256x256_diffusion.pt \
    --data_dir data/my_fine_tune_set \
    --lr 1e-5 \
    --batch_size 4 \
    ...
```

### Multi-GPU Training

```bash
# Using MPI
mpiexec -n 8 python scripts/image_train.py \
    --data_dir data/imagenet \
    --batch_size 32 \  # Per-GPU batch size
    ...

# Effective batch size = 32 × 8 = 256
```

### Gradient Accumulation

For limited GPU memory:

```bash
python scripts/image_train.py \
    --batch_size 32 \      # Desired batch size
    --microbatch 8 \       # Actual per-step batch
    ...  # Accumulates over 4 steps
```

---

## Citation

If you use this codebase in your research, please cite:

```bibtex
@article{dhariwal2021diffusion,
  title={Diffusion Models Beat GANs on Image Synthesis},
  author={Dhariwal, Prafulla and Nichol, Alex},
  journal={arXiv preprint arXiv:2105.05233},
  year={2021}
}
```

---

## License

MIT License - See [LICENSE](LICENSE) file

---

## Additional Resources

- **Paper**: [arXiv:2105.05233](http://arxiv.org/abs/2105.05233)
- **Model Card**: [model-card.md](model-card.md)
- **Original Diffusion Work**: [Ho et al. 2020](https://arxiv.org/abs/2006.11239)
- **DDIM Paper**: [Song et al. 2020](https://arxiv.org/abs/2010.02502)
- **OpenAI Blog**: [Diffusion Models Beat GANs](https://openai.com/blog/diffusion-models-beat-gans/)

---

## Troubleshooting

### Common Issues

**Issue**: Out of memory during training
```bash
# Solutions:
# 1. Reduce batch size
--batch_size 2

# 2. Enable gradient checkpointing
--use_checkpoint True

# 3. Use microbatching
--microbatch 1
```

**Issue**: Slow sampling
```bash
# Use DDIM with fewer steps
--timestep_respacing ddim50 --use_ddim True
```

**Issue**: Poor sample quality
```bash
# 1. Increase guidance scale
--classifier_scale 2.0

# 2. Use more sampling steps
--timestep_respacing 250  # Instead of 50

# 3. Disable clipping if images look oversaturated
--clip_denoised False
```

---

## FAQ

**Q: How long does sampling take?**  
A: With 1000 steps on a V100 GPU: ~30 seconds per image. With DDIM (50 steps): ~1.5 seconds per image.

**Q: Can I use this without a classifier?**  
A: Yes! Use `image_sample.py` instead of `classifier_sample.py`. Quality will be slightly lower.

**Q: What's the difference between this and Stable Diffusion?**  
A: This works in pixel space and came first (2021). Stable Diffusion (2022) works in latent space and is more efficient.

**Q: Can I generate specific objects or scenes?**  
A: With ImageNet models, you can only generate the 1000 ImageNet classes. For arbitrary text prompts, you'd need to integrate CLIP (see "Intended Use" in model card).

**Q: How much training data do I need?**  
A: For good results, typically 10K+ images minimum. ImageNet uses ~1.3M images.

---

*Last updated: October 30, 2025*
