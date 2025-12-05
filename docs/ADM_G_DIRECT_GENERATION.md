# ADM-G: Direct Generation with Classifier Guidance

## Overview

**ADM-G (Ablated Diffusion Model with Guidance)** is OpenAI's direct image generation method that beat BigGAN-deep at 256×256 resolution **without needing upsampling**. This document explains exactly how this generation process works in the codebase.

### Key Achievement
| Method | Resolution | FID ↓ | IS ↑ |
|--------|------------|-------|------|
| BigGAN-deep | 256×256 | 6.95 | 202.6 |
| **ADM-G (This Method)** | 256×256 | **4.59** | **186.7** |

> ADM-G achieves **34% lower FID** than BigGAN-deep using classifier guidance alone!

---

## Block Diagram: ADM-G Generation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ADM-G: DIRECT GENERATION PIPELINE                         │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │  Random Class   │
                              │   y ∈ [0,999]   │
                              │   (ImageNet)    │
                              └────────┬────────┘
                                       │
                                       ▼
┌──────────────┐              ┌─────────────────┐              ┌──────────────┐
│              │              │                 │              │              │
│   Gaussian   │              │  Class Embed    │              │  Timestep    │
│    Noise     │──────────────│  + Time Embed   │──────────────│  Embedding   │
│   x_T ~ N    │              │                 │              │     t        │
│              │              └────────┬────────┘              │              │
└──────┬───────┘                       │                       └──────┬───────┘
       │                               │                              │
       │                               ▼                              │
       │                    ┌─────────────────────┐                   │
       │                    │                     │                   │
       └───────────────────►│      U-Net          │◄──────────────────┘
                            │   (Diffusion Model) │
                            │                     │
                            │  Predicts ε (noise) │
                            │  + variance σ       │
                            └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │                     │
                            │  Predicted x̂₀       │
                            │  (denoised image)   │
                            │                     │
                            └──────────┬──────────┘
                                       │
       ┌───────────────────────────────┼───────────────────────────────┐
       │                               │                               │
       │                               ▼                               │
       │                    ┌─────────────────────┐                   │
       │                    │                     │                   │
       │                    │     Classifier      │                   │
       │                    │   (Noisy Image)     │                   │
       │                    │                     │                   │
       │                    │  p(y | x_t, t)      │                   │
       │                    └──────────┬──────────┘                   │
       │                               │                               │
       │                               ▼                               │
       │                    ┌─────────────────────┐                   │
       │                    │                     │                   │
       │                    │  Compute Gradient   │                   │
       │                    │                     │                   │
       │                    │  ∇_x log p(y|x_t)   │                   │
       │                    │                     │                   │
       │                    └──────────┬──────────┘                   │
       │                               │                               │
       │                               ▼                               │
       │         ┌─────────────────────────────────────────┐          │
       │         │                                         │          │
       │         │     CLASSIFIER GUIDANCE EQUATION        │          │
       │         │                                         │          │
       │         │  μ̃ = μ + s · σ² · ∇_x log p(y|x_t)     │          │
       │         │                                         │          │
       │         │  where s = classifier_scale (e.g., 1.0) │          │
       │         │                                         │          │
       │         └────────────────────┬────────────────────┘          │
       │                              │                                │
       │                              ▼                                │
       │                   ┌─────────────────────┐                    │
       │                   │                     │                    │
       │                   │   Sample x_{t-1}    │                    │
       │                   │                     │                    │
       │                   │  x_{t-1} ~ N(μ̃, σ²) │                    │
       │                   │                     │                    │
       │                   └──────────┬──────────┘                    │
       │                              │                                │
       │                              │                                │
       └──────────────────────────────┼────────────────────────────────┘
                                      │
                                      │  Repeat for t = T, T-1, ..., 1, 0
                                      │
                                      ▼
                           ┌─────────────────────┐
                           │                     │
                           │   Final Image x₀    │
                           │                     │
                           │   256×256 or 512×512│
                           │                     │
                           └─────────────────────┘
```

---

## Mathematical Foundation

### 1. Forward Diffusion Process (Training)

The forward process gradually adds Gaussian noise to an image over $T$ timesteps:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t \mathbf{I})$$

Where $\beta_t$ is the noise schedule. We can sample directly at any timestep $t$:

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$$

Where:
- $\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)$ is the cumulative product
- $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ is Gaussian noise

### 2. Reverse Process (Sampling)

The model learns to reverse this process:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

The U-Net predicts:
- **Noise $\epsilon_\theta$**: Used to compute the mean $\mu_\theta$
- **Variance $\sigma_\theta$**: Learned variance (LEARNED_RANGE in code)

### 3. Classifier Guidance (The Key Innovation!)

Standard sampling produces unconditional images. To generate class-specific images, we modify the sampling using **classifier gradients**:

$$\tilde{\mu} = \mu_\theta(x_t, t) + s \cdot \sigma^2 \cdot \nabla_{x_t} \log p_\phi(y | x_t, t)$$

Where:
- $\mu_\theta$: Predicted mean from diffusion model
- $s$: **Classifier scale** (guidance strength, typically 1.0-4.0)
- $\sigma^2$: Predicted variance
- $\nabla_{x_t} \log p_\phi(y | x_t)$: Gradient from classifier toward class $y$

**This pushes the generation toward images that the classifier recognizes as class $y$!**

---

## Code Files Involved

### File Structure
```
guided-diffusion/
├── guided_diffusion/
│   ├── gaussian_diffusion.py   # Core diffusion math
│   ├── unet.py                 # U-Net architecture
│   ├── script_util.py          # Model creation utilities
│   ├── respace.py              # Timestep respacing
│   └── nn.py                   # Neural network helpers
├── scripts/
│   └── classifier_sample.py    # Official sampling script (MPI)
├── simple_demo.py              # Our simplified sampling (no MPI)
└── models/
    ├── 256x256_diffusion.pt    # Diffusion model weights
    └── 256x256_classifier.pt   # Classifier model weights
```

---

## Detailed Code Walkthrough

### 1. Classifier Guidance Function (`simple_demo.py`, lines 86-94)

```python
def cond_fn(x, t, y=None):
    """
    Compute classifier gradient for guidance.
    
    Args:
        x: Current noisy image x_t
        t: Current timestep
        y: Target class label
    
    Returns:
        Gradient scaled by classifier_scale
    """
    assert y is not None
    with th.enable_grad():
        x_in = x.detach().requires_grad_(True)
        logits = classifier(x_in, t)                    # Classify noisy image
        log_probs = F.log_softmax(logits, dim=-1)       # Log probabilities
        selected = log_probs[range(len(logits)), y.view(-1)]  # Select target class
        return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale
```

**What this does:**
1. Takes the current noisy image $x_t$ and timestep $t$
2. Passes through the classifier to get class predictions
3. Computes $\log p(y | x_t)$ for the target class $y$
4. Computes gradient $\nabla_{x_t} \log p(y | x_t)$ via backpropagation
5. Scales by `classifier_scale` (strength of guidance)

---

### 2. Applying Guidance in Sampling (`gaussian_diffusion.py`, lines 356-369)

```python
def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
    """
    Compute the mean for the previous step, given a function cond_fn that
    computes the gradient of a conditional log probability with respect to x.
    
    This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
    """
    gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
    new_mean = (
        p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
    )
    return new_mean
```

**This implements the key equation:**
$$\tilde{\mu} = \mu + \sigma^2 \cdot \nabla_x \log p(y|x_t)$$

---

### 3. Single Denoising Step (`gaussian_diffusion.py`, lines 394-438)

```python
def p_sample(self, model, x, t, clip_denoised=True, cond_fn=None, model_kwargs=None):
    """
    Sample x_{t-1} from the model at the given timestep.
    """
    # Get predicted mean and variance from U-Net
    out = self.p_mean_variance(
        model, x, t,
        clip_denoised=clip_denoised,
        model_kwargs=model_kwargs,
    )
    
    # Sample noise (zero at t=0)
    noise = th.randn_like(x)
    nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
    
    # Apply classifier guidance if provided
    if cond_fn is not None:
        out["mean"] = self.condition_mean(cond_fn, out, x, t, model_kwargs=model_kwargs)
    
    # Sample x_{t-1} = μ̃ + σ * z
    sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
    return {"sample": sample, "pred_xstart": out["pred_xstart"]}
```

---

### 4. Full Sampling Loop (`gaussian_diffusion.py`, lines 440-475)

```python
def p_sample_loop(self, model, shape, cond_fn=None, model_kwargs=None, progress=False):
    """
    Generate samples from the model.
    """
    # Start from pure noise x_T ~ N(0, I)
    img = th.randn(*shape, device=device)
    
    # Iterate from t=T to t=0
    for t in reversed(range(self.num_timesteps)):
        img = self.p_sample(
            model, img, t,
            cond_fn=cond_fn,           # Classifier guidance!
            model_kwargs=model_kwargs,  # Contains class label y
        )["sample"]
    
    return img  # Final denoised image x_0
```

---

### 5. U-Net Architecture (`unet.py`, lines 427-476)

The diffusion model is a **class-conditional U-Net**:

```python
class UNetModel(nn.Module):
    def __init__(
        self,
        image_size,          # 256 or 512
        in_channels=3,       # RGB
        model_channels=256,  # Base channel width
        out_channels=6,      # 3 for mean + 3 for variance (learn_sigma=True)
        num_res_blocks=2,    # ResBlocks per resolution
        attention_resolutions=(32, 16, 8),  # Where to apply attention
        num_classes=1000,    # ImageNet classes for conditioning
        ...
    ):
        # Time embedding: t → sinusoidal → MLP
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        
        # Class embedding: y → learned embedding
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)
```

**Key architectural features:**
- **Sinusoidal time embeddings**: Encodes timestep $t$ 
- **Class embeddings**: Learned embeddings for each ImageNet class
- **Self-attention at multiple resolutions**: Captures global context
- **ResBlocks with FiLM conditioning**: Uses scale-shift normalization
- **Skip connections**: U-Net encoder-decoder with skip connections

---

### 6. Classifier Architecture (`unet.py`, `EncoderUNetModel`)

The classifier is a **noisy image classifier** trained on noisy images at all timesteps:

```python
class EncoderUNetModel(nn.Module):
    """
    Classifier that takes noisy images and timesteps as input.
    Outputs class logits for 1000 ImageNet classes.
    """
    def __init__(
        self,
        image_size,
        in_channels=3,
        model_channels=128,      # Classifier width
        out_channels=1000,       # ImageNet classes
        num_res_blocks=2,        # Classifier depth
        attention_resolutions=(32, 16, 8),
        pool="attention",        # Attention pooling for classification
        ...
    ):
```

**Why a noisy classifier?**
- Regular classifiers fail on noisy images
- This classifier is trained to recognize classes at **all noise levels**
- Enables accurate gradient computation throughout the denoising process

---

## Key Parameters and Their Effects

### Classifier Scale ($s$)

| Value | Effect |
|-------|--------|
| 0.0 | No guidance (unconditional) |
| 1.0 | Balanced guidance (default) |
| 2.0-4.0 | Strong guidance (better class fidelity) |
| >5.0 | Too strong (artifacts, oversaturation) |

**Trade-off**: Higher scale → better FID, lower diversity

### Noise Schedule

| Schedule | Formula | Use Case |
|----------|---------|----------|
| Linear | $\beta_t \in [0.0001, 0.02]$ | 256×256, 512×512 models |
| Cosine | $\bar{\alpha}_t = \cos^2(\frac{t/T + 0.008}{1.008} \cdot \frac{\pi}{2})$ | 64×64 models |

### Timestep Respacing

Instead of running all 1000 steps, we can resample:
- `timestep_respacing=250`: Use 250 steps (4× faster)
- `timestep_respacing=50`: Use 50 steps (20× faster, some quality loss)

---

## Running ADM-G Generation

### For 256×256 Images

```bash
# First download models
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt -P models/
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt -P models/

# Generate images
python simple_demo.py \
    --model_path models/256x256_diffusion.pt \
    --classifier_path models/256x256_classifier.pt \
    --classifier_scale 1.0 \
    --image_size 256 \
    --num_channels 256 \
    --num_res_blocks 2 \
    --attention_resolutions 32,16,8 \
    --class_cond True \
    --learn_sigma True \
    --noise_schedule linear \
    --diffusion_steps 1000 \
    --timestep_respacing 250 \
    --num_samples 4 \
    --batch_size 1 \
    --classes "207,281,388,130" \
    --output_dir outputs/adm_g_256
```

### For 512×512 Images

```bash
python simple_demo.py \
    --model_path models/512x512_diffusion.pt \
    --classifier_path models/512x512_classifier.pt \
    --classifier_scale 4.0 \
    --image_size 512 \
    --num_channels 256 \
    --num_res_blocks 2 \
    --attention_resolutions 32,16,8 \
    --class_cond True \
    --learn_sigma True \
    --noise_schedule linear \
    --timestep_respacing 250 \
    --num_samples 4 \
    --output_dir outputs/adm_g_512
```

---

## Why ADM-G Beats BigGAN

### 1. Mode Coverage
- **GANs**: Prone to mode collapse (missing diversity)
- **Diffusion**: Covers the full data distribution

### 2. Training Stability
- **GANs**: Adversarial training is unstable
- **Diffusion**: Simple MSE loss on noise prediction

### 3. Classifier Guidance
- Provides precise control over class conditioning
- Gradients directly push samples toward target class
- Can trade off diversity for fidelity (adjustable $s$)

### 4. Learned Variance
- Model predicts both mean AND variance
- Better uncertainty estimation
- Smoother, more realistic outputs

---

## Summary

**ADM-G** achieves state-of-the-art image generation through:

1. **Diffusion Model (U-Net)**: Learns to denoise images step by step
2. **Class Conditioning**: U-Net conditions on ImageNet class embeddings  
3. **Classifier Guidance**: External classifier provides gradients to steer generation
4. **Learned Variance**: Model predicts uncertainty for better sampling

The key insight is that **classifier gradients can guide the diffusion process** without retraining the diffusion model, achieving better results than BigGAN while being more stable to train.

---

## References

- [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) (Dhariwal & Nichol, 2021)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Ho et al., 2020)
- [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) (Nichol & Dhariwal, 2021)
