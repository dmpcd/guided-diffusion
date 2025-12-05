# ADM-G vs ADM-U: Method Comparison

## Overview

This document compares the two main image generation methods from the paper **"Diffusion Models Beat GANs on Image Synthesis"**:

| Method | Full Name | Strategy |
|--------|-----------|----------|
| **ADM-G** | Ablated Diffusion Model with **Guidance** | Direct single-stage generation |
| **ADM-U** | Ablated Diffusion Model with **Upsampling** | Two-stage cascaded generation |

---

## Side-by-Side Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ADM-G vs ADM-U: PIPELINE COMPARISON                       │
└─────────────────────────────────────────────────────────────────────────────┘

          ADM-G (Direct)                      ADM-U (Upsampling)
          ══════════════                      ══════════════════

         ┌─────────────┐                     ┌─────────────┐
         │   Noise     │                     │   Noise     │
         │  512×512    │                     │  128×128    │
         └──────┬──────┘                     └──────┬──────┘
                │                                   │
                ▼                                   ▼
         ┌─────────────┐                     ┌─────────────┐
         │  Diffusion  │                     │  Diffusion  │
         │   Model     │                     │   Model     │
         │  512×512    │                     │  128×128    │
         └──────┬──────┘                     └──────┬──────┘
                │                                   │
                ▼                                   ▼
         ┌─────────────┐                     ┌─────────────┐
         │ Classifier  │                     │ Classifier  │
         │  Guidance   │                     │  Guidance   │
         │   ∇log p    │                     │   ∇log p    │
         └──────┬──────┘                     └──────┬──────┘
                │                                   │
                │                                   ▼
                │                            ┌─────────────┐
                │                            │ Base Image  │
                │                            │  128×128    │
                │                            └──────┬──────┘
                │                                   │
                │                                   ▼
                │                            ┌─────────────┐
                │                            │   Noise     │
                │                            │  512×512    │
                │                            └──────┬──────┘
                │                                   │
                │                                   ▼
                │                            ┌─────────────┐
                │                            │ SuperRes    │
                │                            │   Model     │
                │                            │ [x_t,low]   │
                │                            └──────┬──────┘
                │                                   │
                ▼                                   ▼
         ┌─────────────┐                     ┌─────────────┐
         │   Output    │                     │   Output    │
         │  512×512    │                     │  512×512    │
         │  FID: 7.72  │                     │  FID: 3.85  │
         └─────────────┘                     └─────────────┘

        Single Model                         Two Models
        ~35 seconds                          ~44 seconds
```

---

## Quantitative Comparison

### Image Quality Metrics

| Metric | ADM-G (Direct) | ADM-U (Upsampling) | Winner |
|--------|----------------|--------------------| -------|
| **FID ↓** | 7.72 | **3.85** | ADM-U (2× better) |
| **IS ↑** | 172.7 | **186.2** | ADM-U |
| **Precision ↑** | 0.87 | **0.88** | ADM-U |
| **Recall ↑** | **0.48** | 0.42 | ADM-G |

> **ADM-U produces higher quality images, but ADM-G has better diversity (recall).**

### Computational Cost

| Aspect | ADM-G | ADM-U |
|--------|-------|-------|
| **Models Required** | 2 (diffusion + classifier) | 3 (diffusion + classifier + upsampler) |
| **Model Size** | ~2.1 GB | ~3.3 GB total |
| **GPU Memory** | Higher (512×512 directly) | Lower per stage |
| **Time per Image** | ~35 sec | ~44 sec |
| **Sampling Steps** | 250 | 250 + 250 = 500 total |

### Storage Requirements

| Model | Size | Required For |
|-------|------|--------------|
| `512x512_diffusion.pt` | 2.1 GB | ADM-G only |
| `512x512_classifier.pt` | 208 MB | ADM-G only |
| `128x128_diffusion.pt` | 1.2 GB | ADM-U |
| `128x128_classifier.pt` | 250 MB | ADM-U |
| `128_512_upsampler.pt` | 1.9 GB | ADM-U |

---

## Technical Comparison

### Architecture Differences

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        ARCHITECTURE COMPARISON                              │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ADM-G (Direct 512×512)                                                    │
│  ══════════════════════                                                    │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────┐             │
│  │                    UNetModel                              │             │
│  │  ┌────────────────────────────────────────────────────┐  │             │
│  │  │ Input: 3 channels (RGB)                            │  │             │
│  │  │ Output: 6 channels (mean + variance)               │  │             │
│  │  │ Model channels: 256                                │  │             │
│  │  │ Attention: 32, 16, 8                               │  │             │
│  │  │ Channel mult: (0.5, 1, 1, 2, 2, 4, 4)             │  │             │
│  │  └────────────────────────────────────────────────────┘  │             │
│  └──────────────────────────────────────────────────────────┘             │
│                           +                                                │
│  ┌──────────────────────────────────────────────────────────┐             │
│  │                 EncoderUNetModel (Classifier)             │             │
│  │  ┌────────────────────────────────────────────────────┐  │             │
│  │  │ Input: 3 channels + timestep                       │  │             │
│  │  │ Output: 1000 class logits                          │  │             │
│  │  │ Provides: ∇_x log p(y|x_t)                        │  │             │
│  │  └────────────────────────────────────────────────────┘  │             │
│  └──────────────────────────────────────────────────────────┘             │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ADM-U (128×128 → 512×512)                                                │
│  ═════════════════════════                                                │
│                                                                            │
│  Stage 1: Base Generation                                                  │
│  ┌──────────────────────────────────────────────────────────┐             │
│  │                    UNetModel (128×128)                    │             │
│  │  ┌────────────────────────────────────────────────────┐  │             │
│  │  │ Input: 3 channels                                  │  │             │
│  │  │ Model channels: 256                                │  │             │
│  │  │ Resolution: 128×128                                │  │             │
│  │  └────────────────────────────────────────────────────┘  │             │
│  └──────────────────────────────────────────────────────────┘             │
│                           +                                                │
│  ┌──────────────────────────────────────────────────────────┐             │
│  │              EncoderUNetModel (Classifier 128×128)        │             │
│  └──────────────────────────────────────────────────────────┘             │
│                           ↓                                                │
│  Stage 2: Super-Resolution                                                 │
│  ┌──────────────────────────────────────────────────────────┐             │
│  │                    SuperResModel                          │             │
│  │  ┌────────────────────────────────────────────────────┐  │             │
│  │  │ Input: 6 channels (3 noise + 3 low_res)           │  │             │
│  │  │ Output: 6 channels                                 │  │             │
│  │  │ Model channels: 192                                │  │             │
│  │  │ Conditioning: Concatenation                        │  │             │
│  │  └────────────────────────────────────────────────────┘  │             │
│  └──────────────────────────────────────────────────────────┘             │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### Conditioning Mechanisms

| Aspect | ADM-G | ADM-U Stage 1 | ADM-U Stage 2 |
|--------|-------|---------------|---------------|
| **Class Conditioning** | Embedding | Embedding | Embedding (optional) |
| **Classifier Guidance** | ✅ Yes | ✅ Yes | ❌ No |
| **Low-Res Conditioning** | ❌ No | ❌ No | ✅ Yes (concatenation) |
| **Guidance Equation** | μ̃ = μ + s·σ²·∇log p | μ̃ = μ + s·σ²·∇log p | μ = μ_θ(x_t, low_res) |

---

## Sampling Process Comparison

### ADM-G: Single-Stage Sampling

```python
# Pseudocode for ADM-G
x_T = torch.randn(1, 3, 512, 512)  # Start from noise

for t in reversed(range(T)):
    # 1. Get diffusion model prediction
    eps_pred, sigma_pred = diffusion_model(x_t, t, class_y)
    
    # 2. Compute predicted mean
    mu = compute_mean(x_t, eps_pred, t)
    
    # 3. Get classifier gradient (GUIDANCE!)
    grad = classifier.gradient(x_t, t, class_y)
    
    # 4. Modify mean with guidance
    mu_guided = mu + classifier_scale * sigma² * grad
    
    # 5. Sample x_{t-1}
    x_t = mu_guided + sigma * noise

return x_0  # Final 512×512 image
```

### ADM-U: Two-Stage Sampling

```python
# Pseudocode for ADM-U

# ═══ STAGE 1: Base Generation ═══
x_T = torch.randn(1, 3, 128, 128)  # Start from noise

for t in reversed(range(T)):
    eps_pred, sigma_pred = diffusion_model_128(x_t, t, class_y)
    mu = compute_mean(x_t, eps_pred, t)
    
    # Classifier guidance at 128×128
    grad = classifier_128.gradient(x_t, t, class_y)
    mu_guided = mu + classifier_scale * sigma² * grad
    
    x_t = mu_guided + sigma * noise

base_image = x_0  # 128×128 base

# ═══ STAGE 2: Super-Resolution ═══
low_res = base_image
low_res_upsampled = F.interpolate(low_res, (512, 512), mode='bilinear')

x_T = torch.randn(1, 3, 512, 512)  # Fresh noise at 512×512

for t in reversed(range(T)):
    # Concatenate noisy image with low-res condition
    input = torch.cat([x_t, low_res_upsampled], dim=1)  # 6 channels
    
    eps_pred, sigma_pred = super_res_model(input, t, class_y)
    mu = compute_mean(x_t, eps_pred, t)
    
    # NO classifier guidance here - just low-res conditioning
    x_t = mu + sigma * noise

return x_0  # Final 512×512 image
```

---

## Code Files Comparison

| Purpose | ADM-G Files | ADM-U Files |
|---------|-------------|-------------|
| Main sampling | `simple_demo.py` | `simple_demo.py` + `simple_super_res.py` |
| Diffusion model | `unet.py::UNetModel` | `unet.py::UNetModel` + `unet.py::SuperResModel` |
| Model creation | `script_util.py::create_model_and_diffusion` | `script_util.py::sr_create_model_and_diffusion` |
| Classifier | `unet.py::EncoderUNetModel` | `unet.py::EncoderUNetModel` |
| Core math | `gaussian_diffusion.py` | `gaussian_diffusion.py` (shared) |

---

## ADM-G: File Architecture & Data Flow

### File Structure Overview

```
guided-diffusion/
├── guided_diffusion/
│   ├── gaussian_diffusion.py   # Core diffusion math (forward/reverse process)
│   ├── unet.py                 # U-Net architecture (diffusion + classifier)
│   ├── script_util.py          # Model creation utilities & defaults
│   ├── respace.py              # Timestep respacing (250 → custom steps)
│   └── nn.py                   # Neural network helpers (normalization, etc.)
├── scripts/
│   └── classifier_sample.py    # Official sampling script (requires MPI)
├── simple_demo.py              # Our simplified sampling (no MPI)
└── models/
    ├── 512x512_diffusion.pt    # Diffusion model weights (~2.1 GB)
    └── 512x512_classifier.pt   # Classifier model weights (~208 MB)
```

### Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    ADM-G: FILE INTERACTION ARCHITECTURE                              │
└─────────────────────────────────────────────────────────────────────────────────────┘

                              USER COMMAND
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  simple_demo.py                                                    [ENTRY POINT]    │
│  ═══════════════                                                                    │
│  • Parses command-line arguments (--image_size, --classes, etc.)                   │
│  • Loads model weights from .pt files                                               │
│  • Orchestrates the entire generation pipeline                                      │
│  • Saves output images to disk                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
           │                              │                              │
           │ calls                        │ calls                        │ calls
           ▼                              ▼                              ▼
┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│    script_util.py    │    │    script_util.py    │    │ gaussian_diffusion   │
│ ════════════════════ │    │ ════════════════════ │    │ ════════════════════ │
│                      │    │                      │    │                      │
│ create_model_and_    │    │ create_classifier_   │    │ p_sample_loop()      │
│ diffusion()          │    │ and_diffusion()      │    │                      │
│                      │    │                      │    │ Main sampling loop   │
│ Returns:             │    │ Returns:             │    │ that generates       │
│ • UNetModel          │    │ • EncoderUNetModel   │    │ images step by step  │
│ • GaussianDiffusion  │    │ • GaussianDiffusion  │    │                      │
└──────────────────────┘    └──────────────────────┘    └──────────────────────┘
           │                              │                              │
           │ creates                      │ creates                      │ uses
           ▼                              ▼                              ▼
┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│      unet.py         │    │      unet.py         │    │     respace.py       │
│ ════════════════════ │    │ ════════════════════ │    │ ════════════════════ │
│                      │    │                      │    │                      │
│ class UNetModel      │    │ class EncoderUNet    │    │ SpacedDiffusion      │
│                      │    │ Model                │    │                      │
│ The diffusion model  │    │                      │    │ Allows using fewer   │
│ that predicts noise  │    │ The classifier that  │    │ timesteps (e.g., 250)│
│ ε_θ(x_t, t, y)       │    │ predicts class       │    │ instead of full 1000 │
│                      │    │ p(y|x_t, t)          │    │                      │
└──────────────────────┘    └──────────────────────┘    └──────────────────────┘
           │                              │                              │
           │ uses                         │ uses                         │ wraps
           ▼                              ▼                              ▼
┌──────────────────────┐    ┌──────────────────────────────────────────────────┐
│       nn.py          │    │              gaussian_diffusion.py                │
│ ════════════════════ │    │ ══════════════════════════════════════════════════│
│                      │    │                                                   │
│ Helper functions:    │    │  CORE DIFFUSION MATHEMATICS                       │
│ • normalization()    │    │  ─────────────────────────────                    │
│ • timestep_embedding │    │                                                   │
│ • zero_module()      │    │  Forward:  q(x_t|x_0) = √ᾱ_t x_0 + √(1-ᾱ_t) ε    │
│ • checkpoint()       │    │                                                   │
│                      │    │  Reverse:  p_θ(x_{t-1}|x_t) = N(μ_θ, Σ_θ)        │
│ Low-level building   │    │                                                   │
│ blocks for networks  │    │  With Guidance: μ̃ = μ + s·σ²·∇_x log p(y|x_t)   │
└──────────────────────┘    │                                                   │
                            │  Key Methods:                                     │
                            │  • p_sample() - single denoising step             │
                            │  • p_sample_loop() - full reverse process         │
                            │  • q_sample() - add noise (forward)               │
                            │  • training_losses() - compute loss               │
                            └───────────────────────────────────────────────────┘
                                                   │
                                                   │ loads weights from
                                                   ▼
                            ┌───────────────────────────────────────────────────┐
                            │                    models/                         │
                            │ ══════════════════════════════════════════════════│
                            │                                                   │
                            │  512x512_diffusion.pt   (~2.1 GB)                 │
                            │  ├── UNetModel state_dict                         │
                            │  └── Trained on ImageNet 512×512                  │
                            │                                                   │
                            │  512x512_classifier.pt  (~208 MB)                 │
                            │  ├── EncoderUNetModel state_dict                  │
                            │  └── Noisy classifier for 1000 classes            │
                            │                                                   │
                            └───────────────────────────────────────────────────┘
```

### Detailed Function Call Sequence

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         EXECUTION FLOW: Step by Step                                 │
└─────────────────────────────────────────────────────────────────────────────────────┘

 simple_demo.py                     script_util.py                 gaussian_diffusion.py
 ══════════════                     ══════════════                 ═════════════════════
      │
      │  1. Parse args
      │     (--image_size 512,
      │      --classes "207,281",
      │      --classifier_scale 4.0)
      │
      ├─────────────────────────────────►│
      │  2. create_model_and_diffusion() │
      │                                  │
      │                                  ├──► Creates UNetModel (unet.py)
      │                                  │    with correct architecture
      │                                  │
      │                                  ├──► Creates GaussianDiffusion
      │                                  │    with noise schedule
      │                                  │
      │◄─────────────────────────────────┤
      │  Returns: (model, diffusion)     │
      │
      ├─────────────────────────────────►│
      │  3. create_classifier()          │
      │                                  │
      │                                  ├──► Creates EncoderUNetModel
      │                                  │    (classifier architecture)
      │◄─────────────────────────────────┤
      │  Returns: classifier             │
      │
      │  4. Load weights
      │     model.load_state_dict(
      │       torch.load("512x512_diffusion.pt"))
      │     classifier.load_state_dict(
      │       torch.load("512x512_classifier.pt"))
      │
      │  5. Define cond_fn (guidance function)
      │     ┌────────────────────────────────────────┐
      │     │ def cond_fn(x, t, y):                  │
      │     │   logits = classifier(x, t)           │
      │     │   log_probs = F.log_softmax(logits)   │
      │     │   selected = log_probs[y]             │
      │     │   return torch.autograd.grad(         │
      │     │     selected.sum(), x                 │
      │     │   )[0] * classifier_scale             │
      │     └────────────────────────────────────────┘
      │
      ├──────────────────────────────────────────────────────────────►│
      │  6. diffusion.p_sample_loop_progressive()                     │
      │                                                               │
      │     ┌─────────────────────────────────────────────────────────┤
      │     │  FOR t = T down to 0:                                   │
      │     │                                                         │
      │     │    7a. model(x_t, t, y)                                │
      │     │        └──► UNetModel forward pass                     │
      │     │        └──► Returns: (ε_pred, σ_pred)                  │
      │     │                                                         │
      │     │    7b. Compute μ_θ from ε_pred                         │
      │     │        μ = (x_t - β_t/√(1-ᾱ_t) · ε_pred) / √α_t       │
      │     │                                                         │
      │     │    7c. cond_fn(x_t, t, y)                              │
      │     │        └──► Classifier gradient                        │
      │     │        └──► Returns: ∇_x log p(y|x_t)                  │
      │     │                                                         │
      │     │    7d. Apply guidance                                   │
      │     │        μ̃ = μ + s · σ² · ∇_x log p(y|x_t)              │
      │     │                                                         │
      │     │    7e. Sample x_{t-1}                                   │
      │     │        x_{t-1} = μ̃ + σ · z,  z ~ N(0, I)               │
      │     │                                                         │
      │     └─────────────────────────────────────────────────────────┤
      │                                                               │
      │◄──────────────────────────────────────────────────────────────┤
      │  Returns: x_0 (final image 512×512)                           │
      │
      │  8. Save to outputs/
      │     - sample_0.png
      │     - sample_1.png
      │     - samples_Nx512x512x3.npz
      │
      ▼
   [DONE]
```

### File Responsibilities Summary

| File | Role | Key Functions/Classes |
|------|------|----------------------|
| **simple_demo.py** | 🎯 Entry Point | `main()` - orchestrates everything |
| **script_util.py** | 🏭 Factory | `create_model_and_diffusion()`, `create_classifier()` |
| **gaussian_diffusion.py** | 🧮 Math Engine | `p_sample()`, `p_sample_loop()`, `q_sample()` |
| **unet.py** | 🧠 Neural Networks | `UNetModel`, `EncoderUNetModel` |
| **respace.py** | ⏱️ Time Control | `SpacedDiffusion`, `space_timesteps()` |
| **nn.py** | 🔧 Utilities | `timestep_embedding()`, `normalization()` |
| **models/*.pt** | 💾 Weights | Pre-trained parameters |

### How Each File Contributes to Image Generation

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    WHAT EACH FILE DOES DURING GENERATION                             │
└─────────────────────────────────────────────────────────────────────────────────────┘

   INPUT: Random Noise z ~ N(0,I)           Class Label y = 207 (golden retriever)
                    │                                        │
                    └────────────────────┬───────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  simple_demo.py                                                                      │
│  "I start the process, load models, and save the final image"                       │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  x_T = torch.randn(1, 3, 512, 512)  # I create the initial noise            │   │
│  │  model = load("512x512_diffusion.pt")  # I load the diffusion model         │   │
│  │  classifier = load("512x512_classifier.pt")  # I load the classifier        │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  script_util.py                                                                      │
│  "I know the correct architecture settings for each image size"                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  For 512×512:                                                               │   │
│  │    model_channels = 256                                                     │   │
│  │    attention_resolutions = [32, 16, 8]                                      │   │
│  │    channel_mult = (0.5, 1, 1, 2, 2, 4, 4)                                   │   │
│  │                                                                             │   │
│  │  I create the models with these exact settings!                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  unet.py                                                                             │
│  "I am the neural network brain - I predict noise and classify"                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                             │   │
│  │  UNetModel (Diffusion):              EncoderUNetModel (Classifier):         │   │
│  │  ├── Input: x_t (noisy image)        ├── Input: x_t (noisy image)           │   │
│  │  ├── Condition: t, y                 ├── Condition: t                       │   │
│  │  └── Output: ε_θ (predicted noise)   └── Output: logits for 1000 classes    │   │
│  │                                                                             │   │
│  │  I have encoder + decoder with skip connections and attention!              │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  gaussian_diffusion.py                                                               │
│  "I am the mathematical core - I know how to add/remove noise"                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                             │   │
│  │  p_sample(x_t, t, model_output, cond_fn):                                   │   │
│  │    1. Get ε_θ from model                                                    │   │
│  │    2. Compute μ_θ = f(x_t, ε_θ, α_t, β_t)                                   │   │
│  │    3. Get gradient from classifier: ∇log p(y|x_t)                          │   │
│  │    4. Apply guidance: μ̃ = μ + s·σ²·∇log p(y|x_t)                          │   │
│  │    5. Sample: x_{t-1} = μ̃ + σ·z                                            │   │
│  │                                                                             │   │
│  │  I repeat this 250 times (T → 0) to get the final image!                    │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  respace.py                                                                          │
│  "I control time - I decide which timesteps to use"                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                             │   │
│  │  Original: 1000 timesteps (very slow!)                                      │   │
│  │  Respaced: 250 timesteps (4× faster, almost same quality)                   │   │
│  │                                                                             │   │
│  │  I wrap GaussianDiffusion and adjust the noise schedule accordingly         │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  nn.py                                                                               │
│  "I provide the building blocks for neural networks"                                │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                             │   │
│  │  timestep_embedding(t, dim):  Convert timestep t → embedding vector         │   │
│  │  normalization(channels):     Group normalization layer                     │   │
│  │  zero_module(module):         Initialize output layer to zero               │   │
│  │  checkpoint(fn, *args):       Gradient checkpointing for memory             │   │
│  │                                                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
                            ┌─────────────────────────┐
                            │      OUTPUT IMAGE       │
                            │       512 × 512         │
                            │    Golden Retriever     │
                            └─────────────────────────┘
```

---

## ADM-U: File Architecture & Data Flow

### File Structure Overview

```
guided-diffusion/
├── guided_diffusion/
│   ├── gaussian_diffusion.py   # Core diffusion math (same as ADM-G)
│   ├── unet.py                 # UNetModel + SuperResModel + EncoderUNetModel
│   ├── script_util.py          # Model creation (sr_create_model_and_diffusion)
│   ├── respace.py              # Timestep respacing (used in both stages)
│   └── nn.py                   # Neural network helpers
├── scripts/
│   └── super_res_sample.py     # Official super-res script (requires MPI)
├── simple_demo.py              # Stage 1: Base image generation (no MPI)
├── simple_super_res.py         # Stage 2: Upsampling (no MPI)
└── models/
    ├── 128x128_diffusion.pt    # Base diffusion model (~1.2 GB)
    ├── 128x128_classifier.pt   # Base classifier (~250 MB)
    └── 128_512_upsampler.pt    # Super-resolution model (~1.9 GB)
```

### Complete Two-Stage Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    ADM-U: FILE INTERACTION ARCHITECTURE                              │
│                         (TWO-STAGE PIPELINE)                                         │
└─────────────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════
                              STAGE 1: BASE GENERATION (128×128)
═══════════════════════════════════════════════════════════════════════════════════════

                              USER COMMAND (Stage 1)
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  simple_demo.py                                                    [ENTRY POINT]    │
│  ═══════════════                                                                    │
│  • --image_size 128  (generate at base resolution)                                 │
│  • --classes "207,281,388"  (target classes)                                       │
│  • Loads 128×128 diffusion + classifier models                                     │
│  • Saves base images to outputs/base_128/                                          │
└─────────────────────────────────────────────────────────────────────────────────────┘
           │                              │                              │
           │ calls                        │ calls                        │ calls
           ▼                              ▼                              ▼
┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│    script_util.py    │    │    script_util.py    │    │ gaussian_diffusion   │
│ ════════════════════ │    │ ════════════════════ │    │ ════════════════════ │
│                      │    │                      │    │                      │
│ create_model_and_    │    │ create_classifier_   │    │ p_sample_loop()      │
│ diffusion()          │    │ and_diffusion()      │    │                      │
│                      │    │                      │    │ 250 denoising steps  │
│ For 128×128:         │    │ For 128×128:         │    │ with classifier      │
│ • model_channels=256 │    │ • 1000 class logits  │    │ guidance             │
│ • attention at 32,16 │    │                      │    │                      │
└──────────────────────┘    └──────────────────────┘    └──────────────────────┘
           │                              │                              │
           ▼                              ▼                              ▼
┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│      unet.py         │    │      unet.py         │    │  128x128_diffusion   │
│ ════════════════════ │    │ ════════════════════ │    │       .pt            │
│                      │    │                      │    │ ════════════════════ │
│ class UNetModel      │    │ class EncoderUNet    │    │                      │
│ (128×128 version)    │    │ Model (classifier)   │    │  Pre-trained weights │
│                      │    │                      │    │  for base generation │
│ Predicts ε_θ(x_t,t,y)│    │ Predicts p(y|x_t)    │    │                      │
└──────────────────────┘    └──────────────────────┘    └──────────────────────┘
                                         │
                                         ▼
                            ┌─────────────────────────┐
                            │    BASE IMAGE OUTPUT    │
                            │       128 × 128         │
                            │  samples_3x128x128x3.npz│
                            └────────────┬────────────┘
                                         │
                                         │ saved to disk
                                         ▼

═══════════════════════════════════════════════════════════════════════════════════════
                              STAGE 2: SUPER-RESOLUTION (128→512)
═══════════════════════════════════════════════════════════════════════════════════════

                              USER COMMAND (Stage 2)
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  simple_super_res.py                                               [ENTRY POINT]    │
│  ════════════════════                                                               │
│  • --base_samples outputs/base_128/samples_3x128x128x3.npz                         │
│  • --large_size 512  (target resolution)                                           │
│  • --small_size 128  (input resolution)                                            │
│  • Loads 128→512 upsampler model                                                   │
│  • Saves upsampled images to outputs/upsampled_512/                                │
└─────────────────────────────────────────────────────────────────────────────────────┘
           │                              │                              │
           │ calls                        │ loads                        │ calls
           ▼                              ▼                              ▼
┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│    script_util.py    │    │  Base Image (.npz)   │    │ gaussian_diffusion   │
│ ════════════════════ │    │ ════════════════════ │    │ ════════════════════ │
│                      │    │                      │    │                      │
│ sr_create_model_     │    │ 128×128 images from  │    │ p_sample_loop()      │
│ and_diffusion()      │    │ Stage 1              │    │                      │
│                      │    │                      │    │ 250 denoising steps  │
│ Creates SuperRes     │    │ Bilinear upsampled   │    │ NO classifier        │
│ model configuration  │    │ to 512×512           │    │ guidance here!       │
│                      │    │                      │    │                      │
└──────────────────────┘    └──────────────────────┘    └──────────────────────┘
           │                              │                              │
           │ creates                      │ concatenated                 │ uses
           ▼                              ▼                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│      unet.py :: SuperResModel                                                        │
│ ════════════════════════════════════════════════════════════════════════════════════│
│                                                                                     │
│  ┌───────────────────────────────────────────────────────────────────────────────┐ │
│  │                         SuperResModel Architecture                            │ │
│  │                                                                               │ │
│  │   Input Channels: 6 (3 noise + 3 low_res)     Output Channels: 6             │ │
│  │                                                                               │ │
│  │   ┌─────────────┐    ┌─────────────┐                                         │ │
│  │   │  Noise x_t  │    │  Low-Res    │                                         │ │
│  │   │  512×512×3  │    │  512×512×3  │  (bilinear upsampled from 128)          │ │
│  │   └──────┬──────┘    └──────┬──────┘                                         │ │
│  │          │                  │                                                 │ │
│  │          └────────┬─────────┘                                                 │ │
│  │                   │ CONCATENATE                                               │ │
│  │                   ▼                                                           │ │
│  │          ┌─────────────────┐                                                  │ │
│  │          │  Input Tensor   │                                                  │ │
│  │          │   512×512×6     │                                                  │ │
│  │          └────────┬────────┘                                                  │ │
│  │                   │                                                           │ │
│  │                   ▼                                                           │ │
│  │          ┌─────────────────┐                                                  │ │
│  │          │    U-Net with   │                                                  │ │
│  │          │   model_ch=192  │                                                  │ │
│  │          │  attention@32,16│                                                  │ │
│  │          └────────┬────────┘                                                  │ │
│  │                   │                                                           │ │
│  │                   ▼                                                           │ │
│  │          ┌─────────────────┐                                                  │ │
│  │          │   ε_θ, σ_θ      │  (predicted noise and variance)                  │ │
│  │          │   512×512×6     │                                                  │ │
│  │          └─────────────────┘                                                  │ │
│  │                                                                               │ │
│  │   KEY DIFFERENCE FROM UNetModel:                                             │ │
│  │   • Takes 6 input channels (noise + low_res) instead of 3                    │ │
│  │   • Low-res image provides structural guidance                               │ │
│  │   • No classifier guidance needed - low-res IS the guidance!                 │ │
│  │                                                                               │ │
│  └───────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
                            ┌─────────────────────────┐
                            │   FINAL OUTPUT IMAGE    │
                            │       512 × 512         │
                            │   (Sharp, high quality) │
                            │      FID: 3.85          │
                            └─────────────────────────┘
```

### Detailed Function Call Sequence (Both Stages)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 1: EXECUTION FLOW (Base Generation)                         │
└─────────────────────────────────────────────────────────────────────────────────────┘

 simple_demo.py                     script_util.py                 gaussian_diffusion.py
 ══════════════                     ══════════════                 ═════════════════════
      │
      │  1. Parse args
      │     --image_size 128
      │     --classes "207,281,388"
      │     --classifier_scale 1.0
      │
      ├─────────────────────────────────►│
      │  2. create_model_and_diffusion() │
      │     (for 128×128)                │
      │                                  ├──► Creates UNetModel (128×128)
      │                                  ├──► Creates GaussianDiffusion
      │◄─────────────────────────────────┤
      │
      ├─────────────────────────────────►│
      │  3. create_classifier()          │
      │     (for 128×128)                │
      │◄─────────────────────────────────┤
      │
      │  4. Load 128×128 weights
      │
      ├──────────────────────────────────────────────────────────────►│
      │  5. diffusion.p_sample_loop() with classifier guidance       │
      │                                                               │
      │     FOR t = 250 → 0:                                         │
      │       • model(x_t, t, y) → ε_pred                            │
      │       • classifier(x_t, t) → ∇log p(y|x_t)                  │
      │       • μ̃ = μ + s·σ²·∇log p(y|x_t)                         │
      │       • x_{t-1} = μ̃ + σ·z                                   │
      │◄──────────────────────────────────────────────────────────────┤
      │
      │  6. Save base images
      │     outputs/base_128/samples_3x128x128x3.npz
      │
      ▼
   [STAGE 1 DONE - 128×128 base images saved]


┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 2: EXECUTION FLOW (Super-Resolution)                        │
└─────────────────────────────────────────────────────────────────────────────────────┘

 simple_super_res.py                script_util.py                 gaussian_diffusion.py
 ════════════════════               ══════════════                 ═════════════════════
      │
      │  1. Parse args
      │     --base_samples base_128/samples.npz
      │     --large_size 512
      │     --small_size 128
      │
      │  2. Load base images from .npz
      │     low_res = np.load(...)  # Shape: (N, 128, 128, 3)
      │
      │  3. Upsample low_res to 512×512
      │     low_res_512 = F.interpolate(low_res, (512, 512), mode='bilinear')
      │
      ├─────────────────────────────────►│
      │  4. sr_create_model_and_         │
      │     diffusion()                  │
      │                                  │
      │                                  ├──► Creates SuperResModel
      │                                  │    (input channels = 6)
      │                                  ├──► Creates GaussianDiffusion
      │◄─────────────────────────────────┤
      │
      │  5. Load upsampler weights
      │     model.load("128_512_upsampler.pt")
      │
      │  6. Define model_fn that concatenates low_res
      │     ┌────────────────────────────────────────┐
      │     │ def model_fn(x_t, t, y):               │
      │     │   # x_t: noisy 512×512                 │
      │     │   # low_res_512: upsampled 128→512     │
      │     │   input = torch.cat([x_t, low_res_512])│
      │     │   return model(input, t, y)            │
      │     └────────────────────────────────────────┘
      │
      ├──────────────────────────────────────────────────────────────►│
      │  7. diffusion.p_sample_loop() WITHOUT classifier guidance    │
      │                                                               │
      │     FOR t = 250 → 0:                                         │
      │       • input = [x_t, low_res_512]  (6 channels)             │
      │       • model(input, t, y) → ε_pred                          │
      │       • μ = compute_mean(x_t, ε_pred)                        │
      │       • x_{t-1} = μ + σ·z  (NO gradient guidance!)           │
      │◄──────────────────────────────────────────────────────────────┤
      │
      │  8. Save upsampled images
      │     outputs/upsampled_512/sample_0.png
      │
      ▼
   [STAGE 2 DONE - 512×512 sharp images saved]
```

### File Responsibilities for ADM-U

| File | Stage | Role | Key Functions/Classes |
|------|-------|------|----------------------|
| **simple_demo.py** | 1 | 🎯 Base Generation | Same as ADM-G but at 128×128 |
| **simple_super_res.py** | 2 | 🎯 Upsampling | Loads base, concatenates, upsamples |
| **script_util.py** | Both | 🏭 Factory | `create_model_and_diffusion()`, `sr_create_model_and_diffusion()` |
| **gaussian_diffusion.py** | Both | 🧮 Math Engine | `p_sample_loop()` - same math, different conditioning |
| **unet.py** | 1 | 🧠 Base Model | `UNetModel` (3 input channels) |
| **unet.py** | 2 | 🧠 SuperRes Model | `SuperResModel` (6 input channels) |
| **respace.py** | Both | ⏱️ Time Control | 250 steps each stage = 500 total |

### How Each File Contributes (ADM-U)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    WHAT EACH FILE DOES DURING ADM-U GENERATION                       │
└─────────────────────────────────────────────────────────────────────────────────────┘

═══ STAGE 1 FILES ═══

   INPUT: Random Noise z ~ N(0,I)           Class Label y = 207 (golden retriever)
   at 128×128 resolution                                     │
          │                                                  │
          └──────────────────────┬───────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  simple_demo.py (Stage 1)                                                            │
│  "I generate the base image at low resolution with classifier guidance"            │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  x_T = torch.randn(1, 3, 128, 128)  # Smaller, faster noise                 │   │
│  │  model = load("128x128_diffusion.pt")                                       │   │
│  │  classifier = load("128x128_classifier.pt")                                 │   │
│  │  # Run diffusion with classifier guidance (same as ADM-G)                   │   │
│  │  base_image = diffusion.p_sample_loop(model, cond_fn=classifier_guidance)   │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                   ┌──────────────────────────┐
                   │    BASE IMAGE 128×128    │
                   │  (Good structure, but    │
                   │   low resolution)        │
                   └────────────┬─────────────┘
                                │
                                │ saved as .npz file
                                ▼

═══ STAGE 2 FILES ═══

┌─────────────────────────────────────────────────────────────────────────────────────┐
│  simple_super_res.py (Stage 2)                                                       │
│  "I load the base image and upsample it using diffusion"                            │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  # Load and upsample the low-res image                                      │   │
│  │  low_res = np.load("samples_3x128x128x3.npz")                               │   │
│  │  low_res_512 = F.interpolate(low_res, (512, 512), mode='bilinear')          │   │
│  │                                                                             │   │
│  │  # This blurry upscaled image will GUIDE the diffusion process!            │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  script_util.py :: sr_create_model_and_diffusion()                                   │
│  "I know the special settings for super-resolution models"                          │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  For 128→512 upsampler:                                                     │   │
│  │    in_channels = 6  # 3 noise + 3 low_res (KEY DIFFERENCE!)                │   │
│  │    model_channels = 192                                                     │   │
│  │    num_res_blocks = 2                                                       │   │
│  │    attention_resolutions = [32, 16, 8]                                      │   │
│  │                                                                             │   │
│  │  Returns: (SuperResModel, GaussianDiffusion)                                │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  unet.py :: SuperResModel                                                            │
│  "I am a special U-Net that takes BOTH noise AND low-res as input"                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                             │   │
│  │  class SuperResModel(UNetModel):                                            │   │
│  │      def forward(self, x, timesteps, low_res, **kwargs):                    │   │
│  │          # Concatenate noise with low-res along channel dimension          │   │
│  │          x = torch.cat([x, low_res], dim=1)  # 3 + 3 = 6 channels          │   │
│  │          return super().forward(x, timesteps, **kwargs)                     │   │
│  │                                                                             │   │
│  │  The low-res image tells me WHAT to generate (structure)                    │   │
│  │  The noise provides randomness for DETAILS (textures, fine features)        │   │
│  │                                                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  gaussian_diffusion.py                                                               │
│  "I run the same denoising math, but WITHOUT classifier guidance"                   │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                             │   │
│  │  p_sample(x_t, t, model_output):  # Note: NO cond_fn parameter!            │   │
│  │    1. input = [x_t, low_res_512]  # 6 channels                             │   │
│  │    2. ε_θ = model(input, t)       # Predict noise                          │   │
│  │    3. μ = compute_mean(x_t, ε_θ)  # Compute mean                           │   │
│  │    4. x_{t-1} = μ + σ·z           # Sample (no gradient added!)            │   │
│  │                                                                             │   │
│  │  WHY NO CLASSIFIER?                                                         │   │
│  │  • The low-res image already encodes the class information!                │   │
│  │  • Classifier guidance was used in Stage 1 to create that image            │   │
│  │  • Stage 2 just needs to add high-frequency details                        │   │
│  │                                                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                   ┌──────────────────────────┐
                   │   FINAL IMAGE 512×512    │
                   │  (Sharp details from     │
                   │   diffusion refinement)  │
                   │       FID: 3.85          │
                   └──────────────────────────┘
```

### Key Architectural Difference: UNetModel vs SuperResModel

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         UNetModel vs SuperResModel                                   │
└─────────────────────────────────────────────────────────────────────────────────────┘

  UNetModel (ADM-G, Stage 1)                SuperResModel (Stage 2)
  ══════════════════════════                ════════════════════════
  
  Input: 3 channels                         Input: 6 channels
  ┌─────────────────────┐                   ┌─────────────────────┐
  │     Noise x_t       │                   │ Noise x_t │ Low-Res │
  │      3 × H × W      │                   │   3×H×W   │  3×H×W  │
  └─────────────────────┘                   └─────────────────────┘
           │                                          │
           │                                          │ CONCATENATE
           │                                          │
           ▼                                          ▼
  ┌─────────────────────┐                   ┌─────────────────────┐
  │                     │                   │                     │
  │      Encoder        │                   │      Encoder        │
  │   (downsampling)    │                   │   (sees both!)      │
  │                     │                   │                     │
  └─────────────────────┘                   └─────────────────────┘
           │                                          │
           ▼                                          ▼
  ┌─────────────────────┐                   ┌─────────────────────┐
  │     Bottleneck      │                   │     Bottleneck      │
  │    (attention)      │                   │    (attention)      │
  └─────────────────────┘                   └─────────────────────┘
           │                                          │
           ▼                                          ▼
  ┌─────────────────────┐                   ┌─────────────────────┐
  │                     │                   │                     │
  │      Decoder        │                   │      Decoder        │
  │   (upsampling)      │                   │   (upsampling)      │
  │                     │                   │                     │
  └─────────────────────┘                   └─────────────────────┘
           │                                          │
           ▼                                          ▼
  ┌─────────────────────┐                   ┌─────────────────────┐
  │   Output: ε, σ      │                   │   Output: ε, σ      │
  │      6 × H × W      │                   │      6 × H × W      │
  └─────────────────────┘                   └─────────────────────┘
  
  
  GUIDANCE:                                 GUIDANCE:
  ┌─────────────────────┐                   ┌─────────────────────┐
  │  Classifier adds    │                   │  Low-res image      │
  │  ∇log p(y|x_t)     │                   │  provides structure │
  │  to the mean        │                   │  via concatenation  │
  └─────────────────────┘                   └─────────────────────┘
  
  (External guidance)                       (Built-in guidance)
```

---

## When to Use Each Method

### Use ADM-G (Direct) When:

✅ **Speed is priority** - Single stage is faster  
✅ **Memory is limited** - Don't need to load 3 models  
✅ **Diversity matters** - Higher recall (more varied outputs)  
✅ **Simpler pipeline** - One command, one output  
✅ **Real-time applications** - Lower latency  

### Use ADM-U (Upsampling) When:

✅ **Quality is priority** - 2× better FID  
✅ **Publication/presentation** - Best visual results  
✅ **Fine details matter** - Sharper textures, cleaner edges  
✅ **Faces/complex objects** - More coherent structures  
✅ **Time is available** - Can wait ~25% longer  

---

## Visual Quality Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      EXPECTED VISUAL DIFFERENCES                             │
└─────────────────────────────────────────────────────────────────────────────┘

                    ADM-G (Direct)              ADM-U (Upsampled)
                    ══════════════              ═════════════════

  Global Structure     Good ✓                    Excellent ✓✓
  
  Fine Details         Soft/Blurry               Sharp ✓✓
  
  Textures            Sometimes muddy            Realistic ✓✓
  
  Object Edges         Fuzzy                     Crisp ✓✓
  
  Faces               May have artifacts         More coherent ✓✓
  
  Small Objects        Can be unclear            Well-defined ✓✓
  
  Background           Often noisy               Cleaner ✓✓
  
  Color Consistency    Good ✓                    Excellent ✓✓


  Example (conceptual):
  
  ┌─────────────────────┐     ┌─────────────────────┐
  │  ░░░▓▓▓▓▓░░░░░░░░  │     │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │
  │  ░░▓▓▓▓▓▓▓░░░░░░░  │     │  ▓▓████████████▓▓  │
  │  ░░▓▓███▓▓░░░░░░░  │     │  ▓▓██  ████  ██▓▓  │
  │  ░░▓▓▓▓▓▓▓░░░░░░░  │     │  ▓▓████████████▓▓  │
  │  ░░░▓▓▓▓▓░░░░░░░░  │     │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │
  └─────────────────────┘     └─────────────────────┘
       ADM-G: Softer              ADM-U: Sharper
       edges, less detail         edges, more detail
```

---

## Command Comparison

### ADM-G: Direct 512×512

```bash
python simple_demo.py \
    --model_path models/512x512_diffusion.pt \
    --classifier_path models/512x512_classifier.pt \
    --classifier_scale 4.0 \
    --image_size 512 \
    --classes "207,281,388" \
    --num_samples 3 \
    --output_dir outputs/direct_512
```

### ADM-U: Two-Stage 128→512

```bash
# Stage 1
python simple_demo.py \
    --model_path models/128x128_diffusion.pt \
    --classifier_path models/128x128_classifier.pt \
    --classifier_scale 1.0 \
    --image_size 128 \
    --classes "207,281,388" \
    --num_samples 3 \
    --output_dir outputs/base_128

# Stage 2
python simple_super_res.py \
    --model_path models/128_512_upsampler.pt \
    --base_samples outputs/base_128/samples_3x128x128x3.npz \
    --large_size 512 \
    --small_size 128 \
    --class_cond True \
    --output_dir outputs/upsampled_512
```

---

## The Best of Both Worlds: ADM-G + ADM-U

The paper also combines both methods for optimal results:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ADM-G + ADM-U: COMBINED APPROACH                          │
└─────────────────────────────────────────────────────────────────────────────┘

                         ┌─────────────────┐
                         │     Noise       │
                         │    128×128      │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │   Diffusion     │
                         │  + Classifier   │◄──── ADM-G at 128×128
                         │    Guidance     │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  Base Image     │
                         │   128×128       │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │   SuperRes      │◄──── ADM-U upsampling
                         │    Model        │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │    Output       │
                         │   512×512       │
                         │   FID: 3.94     │
                         └─────────────────┘
```

| Combined Method | FID |
|-----------------|-----|
| ADM-G only (512) | 7.72 |
| ADM-U only (128→512) | 3.85 |
| **ADM-G + ADM-U** | **3.94** |

> Using classifier guidance at 128×128 + upsampling gives near-optimal results!

---

## Summary Table

| Aspect | ADM-G (Direct) | ADM-U (Upsampling) |
|--------|----------------|---------------------|
| **Quality (FID)** | 7.72 | **3.85** ✓ |
| **Diversity (Recall)** | **0.48** ✓ | 0.42 |
| **Speed** | **~35s** ✓ | ~44s |
| **Simplicity** | **Single stage** ✓ | Two stages |
| **Models needed** | **2** ✓ | 3 |
| **Memory per stage** | Higher | **Lower** ✓ |
| **Fine details** | Soft | **Sharp** ✓ |
| **Best for** | Quick demos | **Final outputs** ✓ |

---

## Recommendation

| Scenario | Recommended Method |
|----------|-------------------|
| Quick prototype/demo | ADM-G |
| Presentation to lecturer | **ADM-U** |
| Publication figures | **ADM-U** |
| Real-time application | ADM-G |
| Maximum quality | **ADM-U** or ADM-G + ADM-U |
| Limited GPU memory | **ADM-U** (smaller per-stage) |
| Limited disk space | ADM-G (fewer models) |

---

## References

- [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) - Dhariwal & Nichol, 2021
- [Cascaded Diffusion Models for High Fidelity Image Generation](https://arxiv.org/abs/2106.15282) - Ho et al., 2021
