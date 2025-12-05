# ADM-U: Two-Stage Upsampling Method

## Overview

**ADM-U (Ablated Diffusion Model with Upsampling)** is a two-stage generation pipeline that achieves the **best image quality** in the paper. Instead of generating high-resolution images directly, it first generates low-resolution images and then upsamples them using a separate diffusion model.

### Key Achievement

| Method | Resolution | FID ↓ | Pipeline |
|--------|------------|-------|----------|
| ADM-G (Direct) | 512×512 | 7.72 | Single-stage |
| **ADM-U (Upsampling)** | 128→512 | **3.85** | Two-stage |
| **ADM-G + ADM-U** | 128→512 | **3.94** | Best combined |

> **ADM-U achieves 2× better FID than direct generation!**

---

## Block Diagram: Two-Stage Upsampling Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              ADM-U: TWO-STAGE UPSAMPLING PIPELINE                            │
└─────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
                            STAGE 1: BASE GENERATION
                         (128×128 with Classifier Guidance)
═══════════════════════════════════════════════════════════════════════════════

                              ┌─────────────────┐
                              │  Random Class   │
                              │   y ∈ [0,999]   │
                              │   (ImageNet)    │
                              └────────┬────────┘
                                       │
                                       ▼
┌──────────────┐              ┌─────────────────┐              ┌──────────────┐
│              │              │                 │              │              │
│   Gaussian   │              │    128×128      │              │  Classifier  │
│    Noise     │─────────────►│   Diffusion     │◄─────────────│   Guidance   │
│   x_T ~ N    │              │     Model       │              │  ∇log p(y|x) │
│              │              │                 │              │              │
└──────────────┘              └────────┬────────┘              └──────────────┘
                                       │
                                       │ 250 denoising steps
                                       ▼
                              ┌─────────────────┐
                              │                 │
                              │  Base Image     │
                              │   128×128       │
                              │                 │
                              └────────┬────────┘
                                       │
                                       │
═══════════════════════════════════════════════════════════════════════════════
                            STAGE 2: SUPER-RESOLUTION
                    (128×128 → 512×512 Conditional Upsampling)
═══════════════════════════════════════════════════════════════════════════════
                                       │
                                       ▼
                              ┌─────────────────┐
                              │                 │
                              │   Bilinear      │
                              │   Upsample      │
                              │  128→512        │
                              │                 │
                              └────────┬────────┘
                                       │
                                       │ low_res condition
                                       ▼
┌──────────────┐              ┌─────────────────────────────┐
│              │              │                             │
│   Gaussian   │              │     SuperResModel           │
│    Noise     │─────────────►│   (Conditioned U-Net)       │
│   x_T ~ N    │              │                             │
│  512×512     │              │   input = [x_t, low_res]    │
│              │              │         (6 channels)        │
└──────────────┘              │                             │
                              └──────────────┬──────────────┘
                                             │
                                             │ 250 denoising steps
                                             ▼
                              ┌─────────────────────────────┐
                              │                             │
                              │     High-Resolution         │
                              │        Output               │
                              │       512×512               │
                              │                             │
                              │   (Sharp, detailed,         │
                              │    coherent with base)      │
                              │                             │
                              └─────────────────────────────┘
```

---

## Why Two Stages?

### The Problem with Direct High-Resolution Generation

| Challenge | Direct 512×512 | Two-Stage 128→512 |
|-----------|----------------|-------------------|
| Computational Cost | Very High | Lower per stage |
| Global Coherence | Hard to maintain | Easy at 128×128 |
| Fine Details | Limited by capacity | Dedicated upsampler |
| FID Score | 7.72 | **3.85** |

### The Insight

1. **Stage 1 (128×128)**: Focus on **global structure** - object shape, pose, composition
2. **Stage 2 (Upsampling)**: Focus on **local details** - textures, fine features, sharpness

> It's easier to get the overall structure right at low resolution, then add details!

---

## Mathematical Foundation

### Stage 1: Base Generation (Same as ADM-G)

Standard classifier-guided diffusion at 128×128:

$$x_{t-1} \sim \mathcal{N}(\tilde{\mu}, \sigma^2)$$

Where:
$$\tilde{\mu} = \mu_\theta(x_t, t, y) + s \cdot \sigma^2 \cdot \nabla_{x_t} \log p_\phi(y | x_t)$$

### Stage 2: Conditional Upsampling

The super-resolution model learns:
$$p_\theta(x^{high} | x^{low})$$

Given a low-resolution image $x^{low}$, generate a high-resolution version $x^{high}$.

**Key Difference**: The model is **conditioned** on the low-res image at every step:

$$x_{t-1}^{high} \sim p_\theta(x_{t-1}^{high} | x_t^{high}, x^{low})$$

The conditioning is done by **concatenating** the upsampled low-res image with the noisy high-res image:

$$\text{input} = \text{concat}(x_t^{high}, \text{upsample}(x^{low}))$$

This gives the model 6 input channels instead of 3.

---

## Code Files Involved

### File Structure

```
guided-diffusion/
├── guided_diffusion/
│   ├── gaussian_diffusion.py   # Core diffusion (shared)
│   ├── unet.py                 # Contains SuperResModel
│   ├── script_util.py          # sr_create_model_and_diffusion
│   └── respace.py              # Timestep respacing
├── scripts/
│   └── super_res_sample.py     # Official upsampling (MPI)
├── simple_demo.py              # Stage 1: Base generation
├── simple_super_res.py         # Stage 2: Upsampling (no MPI)
└── models/
    ├── 128x128_diffusion.pt    # Stage 1 diffusion
    ├── 128x128_classifier.pt   # Stage 1 classifier
    └── 128_512_upsampler.pt    # Stage 2 super-res model
```

---

## Detailed Code Walkthrough

### 1. SuperResModel Architecture (`unet.py`, lines 666-680)

```python
class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.
    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, image_size, in_channels, *args, **kwargs):
        # Double the input channels to accept [x_t, low_res]
        super().__init__(image_size, in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        # Upsample low_res to match x's size
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        
        # Concatenate: [noisy_high_res, upsampled_low_res] → 6 channels
        x = th.cat([x, upsampled], dim=1)
        
        return super().forward(x, timesteps, **kwargs)
```

**Key Design:**
- Inherits from standard `UNetModel`
- Takes 6 input channels: 3 (noisy image) + 3 (low-res condition)
- Uses bilinear interpolation to upsample low-res to target size
- Concatenates along channel dimension

---

### 2. Creating Super-Resolution Model (`script_util.py`, lines 280-330)

```python
def sr_create_model_and_diffusion(
    large_size,      # Target size: 512
    small_size,      # Input size: 128
    class_cond,      # Whether to use class conditioning
    learn_sigma,     # Predict variance
    num_channels,    # Base channels: 192
    num_res_blocks,  # ResBlocks: 2
    ...
):
    model = sr_create_model(
        large_size,
        small_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        ...
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        ...
    )
    return model, diffusion


def sr_create_model(large_size, small_size, ...):
    # Channel multipliers for 512 output
    if large_size == 512:
        channel_mult = (1, 1, 2, 2, 4, 4)  # 6 resolution levels
    elif large_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    
    return SuperResModel(
        image_size=large_size,      # 512
        in_channels=3,              # Will become 6 internally
        model_channels=num_channels, # 192
        out_channels=6,             # 3 mean + 3 variance
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_ds,
        channel_mult=channel_mult,
        ...
    )
```

---

### 3. Super-Resolution Sampling (`simple_super_res.py`, lines 60-90)

```python
def main():
    # Load super-resolution model
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(th.load(args.model_path))
    
    # Load base images (128×128)
    data = np.load(args.base_samples)
    base_images = data['arr_0']  # Shape: (N, 128, 128, 3)
    labels = data['arr_1']       # Class labels
    
    # Convert to tensor format
    base_images = th.from_numpy(base_images).permute(0, 3, 1, 2).float()
    base_images = base_images / 127.5 - 1.0  # Normalize to [-1, 1]
    
    for i in range(len(base_images)):
        # Get low-res condition
        low_res = base_images[i:i+1].to(device)
        
        # Model kwargs include low_res and class label
        model_kwargs = {"low_res": low_res}
        if args.class_cond:
            model_kwargs["y"] = th.tensor([labels[i]], device=device)
        
        # Standard diffusion sampling, but conditioned on low_res
        sample = diffusion.p_sample_loop(
            model,
            (1, 3, args.large_size, args.large_size),  # 512×512 output
            model_kwargs=model_kwargs,
            progress=True,
        )
```

---

### 4. Low-Res Data Loading (`super_res_sample.py`, lines 74-95)

```python
def load_data_for_worker(base_samples, batch_size, class_cond):
    """Load and prepare low-resolution images for upsampling."""
    
    with bf.BlobFile(base_samples, "rb") as f:
        obj = np.load(f)
        image_arr = obj["arr_0"]   # Low-res images
        if class_cond:
            label_arr = obj["arr_1"]  # Class labels
    
    while True:
        for i in range(len(image_arr)):
            buffer.append(image_arr[i])
            if class_cond:
                label_buffer.append(label_arr[i])
            
            if len(buffer) == batch_size:
                # Prepare batch
                batch = th.from_numpy(np.stack(buffer)).float()
                batch = batch / 127.5 - 1.0      # Normalize
                batch = batch.permute(0, 3, 1, 2)  # BHWC → BCHW
                
                res = {"low_res": batch}
                if class_cond:
                    res["y"] = th.from_numpy(np.stack(label_buffer))
                
                yield res
```

---

## Available Upsampling Pipelines

### Pipeline 1: 64×64 → 256×256

```bash
# Stage 1: Generate 64×64 base images
python simple_demo.py \
    --model_path models/64x64_diffusion.pt \
    --classifier_path models/64x64_classifier.pt \
    --image_size 64 \
    --num_samples 4 \
    --output_dir outputs/base_64

# Stage 2: Upsample to 256×256
python simple_super_res.py \
    --model_path models/64_256_upsampler.pt \
    --base_samples outputs/base_64/samples_4x64x64x3.npz \
    --large_size 256 \
    --small_size 64 \
    --output_dir outputs/upsampled_256
```

### Pipeline 2: 128×128 → 512×512 (Best Quality)

```bash
# Stage 1: Generate 128×128 base images
python simple_demo.py \
    --model_path models/128x128_diffusion.pt \
    --classifier_path models/128x128_classifier.pt \
    --classifier_scale 1.0 \
    --image_size 128 \
    --num_samples 4 \
    --classes "207,281,388,130" \
    --output_dir outputs/base_128

# Stage 2: Upsample to 512×512
python simple_super_res.py \
    --model_path models/128_512_upsampler.pt \
    --base_samples outputs/base_128/samples_4x128x128x3.npz \
    --large_size 512 \
    --small_size 128 \
    --class_cond True \
    --output_dir outputs/upsampled_512
```

---

## Key Parameters

### Stage 1 (Base Generation)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `image_size` | 128 | Base resolution |
| `classifier_scale` | 1.0 | Guidance strength |
| `num_channels` | 256 | Model capacity |
| `noise_schedule` | linear | Noise scheduling |
| `timestep_respacing` | 250 | Sampling steps |

### Stage 2 (Super-Resolution)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `large_size` | 512 | Target resolution |
| `small_size` | 128 | Input resolution |
| `num_channels` | 192 | Model capacity |
| `class_cond` | True | Use class labels |
| `timestep_respacing` | 250 | Sampling steps |

---

## Architecture Comparison

### Base Model vs Super-Resolution Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ARCHITECTURE COMPARISON                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────┐    ┌─────────────────────────────┐
│     BASE DIFFUSION MODEL    │    │    SUPER-RES MODEL          │
│       (Stage 1)             │    │       (Stage 2)             │
├─────────────────────────────┤    ├─────────────────────────────┤
│                             │    │                             │
│  Input: 3 channels          │    │  Input: 6 channels          │
│         (RGB noise)         │    │    (3 noise + 3 low_res)    │
│                             │    │                             │
│  Conditioning:              │    │  Conditioning:              │
│  - Timestep t               │    │  - Timestep t               │
│  - Class label y            │    │  - Class label y            │
│                             │    │  - Low-res image (concat)   │
│                             │    │                             │
│  Output: 3 or 6 channels    │    │  Output: 6 channels         │
│         (mean, [variance])  │    │         (mean + variance)   │
│                             │    │                             │
│  Model channels: 256        │    │  Model channels: 192        │
│  Resolution: 128×128        │    │  Resolution: 512×512        │
│                             │    │                             │
└─────────────────────────────┘    └─────────────────────────────┘
         │                                    │
         │                                    │
         ▼                                    ▼
┌─────────────────────────────┐    ┌─────────────────────────────┐
│                             │    │                             │
│  Has CLASSIFIER GUIDANCE    │    │  NO classifier guidance     │
│  (separate classifier net)  │    │  (just low_res condition)   │
│                             │    │                             │
└─────────────────────────────┘    └─────────────────────────────┘
```

---

## Training the Super-Resolution Model (Reference)

The upsampler is trained differently from the base model:

```python
# Training objective: predict noise given low_res condition
# 
# 1. Sample (x_high, x_low, y) from training data
# 2. Sample timestep t and noise ε
# 3. Create noisy image: x_t = √ᾱ_t * x_high + √(1-ᾱ_t) * ε  
# 4. Concatenate: input = [x_t, upsample(x_low)]
# 5. Predict: ε_θ = model(input, t, y)
# 6. Loss: ||ε - ε_θ||²
```

**Key Training Details:**
- Uses real (x_high, x_low) pairs from ImageNet
- Low-res images are downsampled from high-res originals
- Model learns to reconstruct high-frequency details
- Class conditioning helps maintain semantic consistency

---

## Timing Comparison

| Method | Time per Image (RTX 4090) |
|--------|---------------------------|
| Direct 512×512 | ~35 seconds |
| Stage 1 (128×128) | ~7 seconds |
| Stage 2 (128→512) | ~37 seconds |
| **Total Two-Stage** | ~44 seconds |

> Slightly slower, but **2× better quality!**

---

## Quality Comparison

### Expected Visual Differences

| Aspect | Direct 512×512 | Upsampled 512×512 |
|--------|----------------|-------------------|
| Global Structure | Good | Excellent |
| Fine Details | Soft/Blurry | Sharp |
| Textures | Sometimes muddy | Realistic |
| Faces | May have artifacts | More coherent |
| Object Edges | Fuzzy | Crisp |

---

## Complete Pipeline Script

```bash
#!/bin/bash
# generate_superres_512.sh - Two-stage 128→512 generation

# Stage 1: Base images
python simple_demo.py \
    --model_path models/128x128_diffusion.pt \
    --classifier_path models/128x128_classifier.pt \
    --classifier_scale 1.0 \
    --image_size 128 \
    --classes "9,130,263,1" \
    --num_samples 4 \
    --output_dir outputs/base_128

# Stage 2: Upsample
python simple_super_res.py \
    --model_path models/128_512_upsampler.pt \
    --base_samples outputs/base_128/samples_4x128x128x3.npz \
    --large_size 512 \
    --small_size 128 \
    --class_cond True \
    --output_dir outputs/upsampled_512
```

---

## Summary

**ADM-U (Two-Stage Upsampling)** achieves the best image quality by:

1. **Stage 1**: Generate 128×128 images with classifier guidance
   - Focus on global structure and composition
   - Classifier ensures correct class appearance
   
2. **Stage 2**: Upsample 128→512 with conditional diffusion
   - Low-res image provides structure constraint
   - Model adds high-frequency details and textures
   - Result: Sharp, coherent, realistic images

**Key Innovation**: The `SuperResModel` concatenates the low-res condition with the noisy image, giving the model explicit information about what structure to preserve while adding details.

---

## References

- [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) (Dhariwal & Nichol, 2021)
- [Cascaded Diffusion Models](https://arxiv.org/abs/2106.15282) (Ho et al., 2021)
- [Image Super-Resolution via Iterative Refinement](https://arxiv.org/abs/2104.07636) (SR3, Saharia et al., 2021)
