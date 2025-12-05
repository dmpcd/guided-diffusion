# Enhanced Prompt for Claude Sonnet 4.5 Agent: Implement ICG + TSG in Guided-Diffusion

## Context

I'm working with the OpenAI guided-diffusion repository (https://github.com/openai/guided-diffusion) which implements "Diffusion Models Beat GANs on Image Synthesis" (Dhariwal & Nichol, 2021). I want to add **Independent Condition Guidance (ICG) AND Timestep Guidance (TSG)** from the paper "No Training, No Problem: Rethinking Classifier-Free Guidance for Diffusion Models" (Sadat et al., ICLR 2025) to improve sampling quality.

## Why Both ICG and TSG?

- **ICG**: Training-free guidance for conditional models (replaces classifier guidance)
- **TSG**: Training-free guidance for BOTH conditional AND unconditional models
- **Together**: Can be combined for even better results (as shown in Table 4 of the paper)
- **Benefits**: No auxiliary model training, no classifier needed, same computational cost as CFG

## Your Task

Implement BOTH Independent Condition Guidance (ICG) and Timestep Guidance (TSG) in the guided-diffusion codebase. Both are **inference-only modifications**.

---

# PART 1: IMPLEMENT INDEPENDENT CONDITION GUIDANCE (ICG)

## 1. Create `scripts/icg_sample.py`

Create a new sampling script based on `scripts/classifier_sample.py` with these modifications:

**Key Changes:**
- Remove all classifier-related code (no classifier loading, no classifier model)
- Implement ICG sampling loop that uses the diffusion model only
- Add `icg_scale` parameter (default: 1.5)
- Generate samples using the new `p_sample_loop_icg()` method

**Script Structure:**
```python
# Import standard modules and guided_diffusion utilities
# Load ONLY the diffusion model (no classifier)
# Implement sampling loop that:
#   - Generates class labels
#   - Calls diffusion.p_sample_loop_icg() with icg_scale parameter
#   - Saves results to .npz file
# Add argparser with icg_scale, num_samples, batch_size, etc.
```

## 2. Add ICG Method to `guided_diffusion/gaussian_diffusion.py`

Add a new method `p_sample_loop_icg()` to the `GaussianDiffusion` class.

**Method Signature:**
```python
def p_sample_loop_icg(
    self,
    model,
    shape,
    noise=None,
    clip_denoised=True,
    denoised_fn=None,
    model_kwargs=None,
    device=None,
    progress=False,
    icg_scale=1.5,
):
    """
    Generate samples from the model using Independent Condition Guidance.
    
    This is a training-free alternative to classifier guidance that achieves
    similar quality without requiring an auxiliary classifier model.
    
    Args:
        model: The conditional diffusion model
        shape: Tuple specifying output shape (batch_size, channels, height, width)
        noise: Optional initial noise, if None will sample from N(0,I)
        clip_denoised: Whether to clip denoised values to [-1, 1]
        denoised_fn: Optional function to apply to denoised samples
        model_kwargs: Dict containing 'y' (class labels or conditions)
        device: Device to run on
        progress: Whether to show tqdm progress bar
        icg_scale: Guidance scale (default: 1.5 for ImageNet 256×256)
    
    Returns:
        Tensor of generated samples with shape `shape`
    """
```

**ICG Algorithm:**
```
For each reverse diffusion timestep t (from T to 0):
    1. Get target condition: y_cond (the class we want to generate)
    
    2. Generate random independent condition: y_random
       - Option A (preferred): Random class label from [0, num_classes)
       - Option B: Gaussian noise with same statistics as y_cond
    
    3. Get conditional prediction:
       out_cond = model(x_t, t, y_cond)
    
    4. Get "unconditional" prediction using random condition:
       out_uncond = model(x_t, t, y_random)
    
    5. Apply ICG guidance formula:
       mean_guided = out_uncond["mean"] + icg_scale * (out_cond["mean"] - out_uncond["mean"])
    
    6. Use mean_guided with out_cond["variance"] for sampling step
    
    7. Sample next x_{t-1} using reparameterization trick
```

**ICG Formula:**
```
D̂(z_t, t, y) = D_θ(z_t, t, ŷ) + w · (D_θ(z_t, t, y) - D_θ(z_t, t, ŷ))

Where:
- y: Target condition (desired class label)
- ŷ: Random independent condition (random class label)
- w: Guidance scale (icg_scale)
- D_θ: Denoising prediction from the model
```

---

# PART 2: IMPLEMENT TIMESTEP GUIDANCE (TSG)

## NEW: Add TSG Method to `guided_diffusion/gaussian_diffusion.py`

TSG is a new guidance method that works on BOTH conditional AND unconditional models by perturbing the timestep embedding.

**Method Signature:**
```python
def p_sample_loop_tsg(
    self,
    model,
    shape,
    noise=None,
    clip_denoised=True,
    denoised_fn=None,
    model_kwargs=None,
    device=None,
    progress=False,
    tsg_scale=2.5,
    tsg_s=2.0,
    tsg_alpha=1.0,
    tsg_t_min=0,
    tsg_t_max=1000,
):
    """
    Generate samples using Timestep Guidance (TSG).
    
    TSG is a training-free guidance method that works on both conditional
    and unconditional models by perturbing the timestep embedding.
    
    Args:
        model: The diffusion model (conditional or unconditional)
        shape: Tuple specifying output shape (batch_size, channels, height, width)
        noise: Optional initial noise, if None will sample from N(0,I)
        clip_denoised: Whether to clip denoised values to [-1, 1]
        denoised_fn: Optional function to apply to denoised samples
        model_kwargs: Dict containing optional 'y' (class labels/text)
        device: Device to run on
        progress: Whether to show tqdm progress bar
        tsg_scale: Guidance scale (default: 2.5, range: 2.5-4.0 for conditional)
        tsg_s: Noise scale parameter (default: 2.0)
        tsg_alpha: Power schedule exponent (default: 1.0)
        tsg_t_min: Minimum timestep to apply TSG (default: 0)
        tsg_t_max: Maximum timestep to apply TSG (default: 1000)
    
    Returns:
        Tensor of generated samples with shape `shape`
    """
```

**TSG Algorithm:**
```
For each reverse diffusion timestep t (from T to 0):
    1. Get normal timestep embedding: t_emb
    
    2. Perturb timestep embedding using noise schedule:
       noise_scale = tsg_s * (t ^ tsg_alpha) * std(t_emb)
       t_emb_perturbed = t_emb + noise_scale * Gaussian_noise
    
    3. Get model output with normal timestep:
       out_normal = model(x_t, t_emb, y)
    
    4. Get model output with perturbed timestep:
       out_perturbed = model(x_t, t_emb_perturbed, y)
    
    5. Apply TSG guidance formula:
       mean_guided = out_perturbed["mean"] + tsg_scale * (out_normal["mean"] - out_perturbed["mean"])
    
    6. Use mean_guided with out_normal["variance"] for sampling step
    
    7. Sample next x_{t-1} using reparameterization trick
```

**TSG Formula:**
```
D̂(z_t, t) = D_θ(z_t, t̃) + w_TSG · (D_θ(z_t, t) - D_θ(z_t, t̃))

Where:
- t: Normal timestep embedding
- t̃: Perturbed timestep embedding
  - t̃_emb = t_emb + s·t^α·n, where n ∼ N(0,I)
- w_TSG: Guidance scale (tsg_scale, typically 2.5-4.0)
- D_θ: Denoising prediction from the model
```

**TSG Hyperparameters (from Paper Table 11):**

For **Unconditional Models** (DiT-XL/2):
```python
tsg_scale=5.0
tsg_s=1.0
tsg_alpha=1.0
tsg_t_min=200
tsg_t_max=800
```

For **Conditional Models** (DiT-XL/2 conditional):
```python
tsg_scale=2.5
tsg_s=2.0
tsg_alpha=1.0
tsg_t_min=0
tsg_t_max=1000
```

For **Text-to-Image Unconditional** (Stable Diffusion):
```python
tsg_scale=3.0
tsg_s=1.25
tsg_alpha=1.0
tsg_t_min=100
tsg_t_max=900
```

For **Text-to-Image Conditional** (Stable Diffusion):
```python
tsg_scale=4.0
tsg_s=3.0
tsg_alpha=0.25
tsg_t_min=400
tsg_t_max=1000
```

---

## 3. Create `scripts/tsg_sample.py` (NEW)

Create a NEW sampling script for TSG-only guidance:

**Key Features:**
- Works on BOTH conditional and unconditional models
- No condition information required for sampling
- Add `--tsg_scale`, `--tsg_s`, `--tsg_alpha` parameters
- Can be used without providing class labels
- Perfect for unconditional model testing

**Script Structure:**
```python
# Load diffusion model (no classifier, no special conditioning)
# Can work with or without conditional model
# Implements TSG sampling loop
# Saves results to .npz file
# Add argparser with TSG-specific parameters
```

---

## 4. Create `scripts/icg_tsg_combined_sample.py` (BONUS)

Optionally create a COMBINED sampling script that uses both ICG and TSG together:

**From Paper Table 4:**
```
- Without guidance: FID = 15.49
- ICG only: FID = 6.47
- TSG only: FID = 9.55
- ICG + TSG combined: FID = 5.76 (BEST!)
```

**Benefits of combining:**
- Better FID than either method alone
- Better precision (quality)
- Good precision/recall balance
- Complementary guidance signals

**Implementation:**
```python
# Step 1: Get ICG guided prediction
mean_icg = out_uncond_cond["mean"] + icg_scale * (out_cond["mean"] - out_uncond_cond["mean"])

# Step 2: Apply TSG to both conditional branch
out_normal = model(x_t, t_emb, y)
out_perturbed = model(x_t, t_emb_perturbed, y)
mean_tsg = out_perturbed["mean"] + tsg_scale * (out_normal["mean"] - out_perturbed["mean"])

# Step 3: Combine guidance signals (or apply sequentially)
# Option A: Average the means
mean_combined = (mean_icg + mean_tsg) / 2

# Option B: Apply ICG, then TSG on the result
```

---

# TSG SPECIAL CONSIDERATIONS

## Why TSG Works

From the paper, TSG leverages Langevin dynamics:
- Lower timesteps cause **excessive noise removal** (soft, blurry outputs)
- Higher timesteps cause **insufficient noise removal** (noisy outputs)
- TSG perturbs timestep to explore both, then uses their difference to guide toward better solutions

## Noise Schedule Options

**Constant Schedule:**
```python
noise_scale = s
t_emb_perturbed = t_emb + noise_scale * randn_like(t_emb)
```

**Power Schedule (Recommended):**
```python
noise_scale = s * (t ^ alpha)  # where t is normalized to [0,1]
t_emb_perturbed = t_emb + noise_scale * t_emb.std() * randn_like(t_emb)
```

## Layer-Wise Application

From paper appendix: TSG can be applied selectively to layers:
- Apply to first N layers of UNet/Transformer
- Skip application to final layers
- Recommended: Apply to first 10 layers

Implementation:
```python
# Only perturb timestep embedding for first 10 layers
if layer_index < 10:
    use_perturbed_temb = True
else:
    use_perturbed_temb = False
```

---

# IMPLEMENTATION PRIORITIES

## Priority 1 (Essential)
1. ✅ Add `p_sample_loop_icg()` to gaussian_diffusion.py
2. ✅ Create `scripts/icg_sample.py`
3. ✅ Test ICG with conditional model

## Priority 2 (Highly Recommended)
4. ✅ Add `p_sample_loop_tsg()` to gaussian_diffusion.py
5. ✅ Create `scripts/tsg_sample.py`
6. ✅ Test TSG with unconditional model

## Priority 3 (Bonus)
7. ⭐ Create `scripts/icg_tsg_combined_sample.py` (uses both methods)
8. ⭐ Add layer-wise TSG application
9. ⭐ Add power schedule vs constant schedule option

---

# TESTING COMMANDS

**Test ICG:**
```bash
python scripts/icg_sample.py \
    --model_path models/256x256_diffusion.pt \
    --attention_resolutions 32,16,8 \
    --class_cond True \
    --diffusion_steps 1000 \
    --image_size 256 \
    --learn_sigma True \
    --noise_schedule linear \
    --num_channels 256 \
    --num_head_channels 64 \
    --num_res_blocks 2 \
    --resblock_updown True \
    --use_fp16 True \
    --use_scale_shift_norm True \
    --icg_scale 1.5 \
    --batch_size 4 \
    --num_samples 16
```

**Test TSG (conditional):**
```bash
python scripts/tsg_sample.py \
    --model_path models/256x256_diffusion.pt \
    --attention_resolutions 32,16,8 \
    --class_cond True \
    --diffusion_steps 1000 \
    --image_size 256 \
    --learn_sigma True \
    --noise_schedule linear \
    --num_channels 256 \
    --num_head_channels 64 \
    --num_res_blocks 2 \
    --resblock_updown True \
    --use_fp16 True \
    --use_scale_shift_norm True \
    --tsg_scale 2.5 \
    --tsg_s 2.0 \
    --tsg_alpha 1.0 \
    --batch_size 4 \
    --num_samples 16
```

**Test Combined (ICG + TSG):**
```bash
python scripts/icg_tsg_combined_sample.py \
    --model_path models/256x256_diffusion.pt \
    --attention_resolutions 32,16,8 \
    --class_cond True \
    --icg_scale 1.4 \
    --tsg_scale 2.5 \
    --batch_size 4 \
    --num_samples 16
```

---

# EXPECTED RESULTS (from Paper)

| Method | Model | FID ↓ | Precision ↑ | Recall ↑ |
|--------|-------|-------|------------|---------|
| Baseline | DiT-XL/2 | 15.49 | 0.64 | 0.74 |
| ICG only | DiT-XL/2 | 6.47 | 0.77 | 0.69 |
| TSG only | DiT-XL/2 | 9.55 | 0.70 | 0.71 |
| **ICG + TSG** | **DiT-XL/2** | **5.76** | **0.82** | **0.65** |

---

# DELIVERABLES

Please provide:

1. **`scripts/icg_sample.py`** - Complete ICG sampling script
2. **`scripts/tsg_sample.py`** - Complete TSG sampling script (NEW)
3. **Modified `gaussian_diffusion.py`** with:
   - `p_sample_loop_icg()` method
   - `p_sample_loop_tsg()` method (NEW)
4. **Optional `scripts/icg_tsg_combined_sample.py`** - Combined guidance script
5. **Usage examples and comparison** showing:
   - ICG performance vs baseline
   - TSG performance vs baseline
   - ICG + TSG performance (best)

---

# CODE QUALITY CHECKLIST

- ✅ Follow existing code style from guided-diffusion
- ✅ Add comprehensive docstrings
- ✅ Use type hints where applicable
- ✅ Handle edge cases (None values, device mismatches)
- ✅ Add clear comments explaining TSG/ICG logic
- ✅ Support both PyTorch FP32 and FP16
- ✅ Work with batched and single sample inference
- ✅ No breaking changes to existing functionality

---

This comprehensive prompt will enable GitHub Copilot to implement both ICG and TSG for you. The key is that both are pure inference-time modifications requiring no model retraining!
