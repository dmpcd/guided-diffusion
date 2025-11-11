# Prompt for Claude Sonnet 4.5 Agent: Implement ICG in Guided-Diffusion

## Context

I'm working with the OpenAI guided-diffusion repository (https://github.com/openai/guided-diffusion) which implements "Diffusion Models Beat GANs on Image Synthesis" (Dhariwal & Nichol, 2021). I want to add Independent Condition Guidance (ICG) from the paper "No Training, No Problem: Rethinking Classifier-Free Guidance for Diffusion Models" (Sadat et al., ICLR 2025) to improve the sampling quality without requiring a classifier.

## Your Task

Implement Independent Condition Guidance (ICG) in the guided-diffusion codebase. This is an **inference-only modification** that eliminates the need for classifier guidance while maintaining or improving sample quality.

## What You Need to Implement

### 1. Create `scripts/icg_sample.py`

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

### 2. Add ICG Method to `guided_diffusion/gaussian_diffusion.py`

Add a new method `p_sample_loop_icg()` to the `GaussianDiffusion` class that implements the ICG sampling algorithm.

**ICG Algorithm (Pseudocode):**
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
        icg_scale: Guidance scale (higher = better quality, less diversity)
                   Recommended: 1.4-1.5 for ImageNet 128/256, 3.0 for text-to-image
    
    Returns:
        Tensor of generated samples with shape `shape`
    """
```

**Implementation Details:**
- Use `self.p_mean_variance()` to get model predictions (already exists in the codebase)
- For random condition, use: `torch.randint(0, 1000, size=y_cond.shape, device=device)` for ImageNet
- Apply guidance to the mean prediction, not the variance
- Use variance from the conditional prediction (`out_cond["variance"]`)
- Follow the same sampling pattern as existing `p_sample_loop()` method
- Support progress bar with tqdm if `progress=True`

### 3. Optional: Add Command-line Interface

If creating a standalone script, add argparse with these parameters:
- `--icg_scale`: Guidance scale (default: 1.5)
- `--num_samples`: Number of samples to generate (default: 10000)
- `--batch_size`: Batch size (default: 16)
- `--model_path`: Path to pretrained diffusion model (required)
- `--num_classes`: Number of classes (default: 1000 for ImageNet)
- `--clip_denoised`: Whether to clip denoised values (default: True)
- All standard model configuration flags from `model_and_diffusion_defaults()`

## Important Implementation Notes

### ICG Core Formula
The key equation you're implementing is:

**D̂(z_t, t, y) = D_θ(z_t, t, ŷ) + w · (D_θ(z_t, t, y) - D_θ(z_t, t, ŷ))**

Where:
- `y`: Target condition (desired class label)
- `ŷ`: Random independent condition (random class label)
- `w`: Guidance scale (`icg_scale`)
- `D_θ`: Denoising prediction from the model

This can be rewritten as:
**D̂ = (1 - w) · D_uncond + w · D_cond**

### Random Condition Generation

**For class-conditional models (ImageNet):**
```python
# Preferred approach
y_random = torch.randint(
    low=0, 
    high=1000,  # num_classes
    size=y_cond.shape, 
    device=device
)
```

**For embedding-based conditioning (text-to-image):**
```python
# Alternative approach
y_random = torch.randn_like(y_cond.float()) * y_cond.float().std()
```

### Differences from Classifier Guidance

**What to REMOVE:**
- ❌ Classifier model loading
- ❌ `cond_fn` function that computes classifier gradients
- ❌ Gradient computation (`torch.enable_grad()` context)
- ❌ Any reference to `classifier_scale`

**What to ADD:**
- ✅ Random condition sampling at each timestep
- ✅ Second forward pass with random condition
- ✅ ICG guidance formula combining both predictions
- ✅ `icg_scale` parameter

### Code Structure Reference

Look at these existing methods for reference:
- `p_sample_loop()` - Basic sampling loop structure
- `p_sample_loop_progressive()` - Progressive sampling with yields
- `p_mean_variance()` - Getting model predictions
- `p_sample()` - Single sampling step

Model your `p_sample_loop_icg()` after `p_sample_loop()` but with the ICG modifications.

## Testing Your Implementation

### 1. Basic Test
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

### 2. Verify Output
- Script should complete without errors
- Output should be saved as `.npz` file
- File should contain image array and label array
- Images should be properly formatted (batch_size, 3, 256, 256)

### 3. Visual Inspection
- Load generated images and verify they look reasonable
- Compare quality with classifier guidance output (if available)
- Check that images match their class labels

## Expected Behavior

**Computational Cost:**
- ICG requires 2 forward passes per timestep (same as classifier guidance)
- One pass with target condition `y`
- One pass with random condition `ŷ`
- No gradient computation needed (faster than classifier guidance)

**Performance Targets:**
Based on the paper, you should achieve:
- FID similar to or better than classifier guidance
- Better precision (quality) with same or similar recall (diversity)
- ~30% faster training if retraining (not applicable here, but good to know)

**Hyperparameter Recommendations:**
- ImageNet 128×128: `icg_scale = 1.4`
- ImageNet 256×256: `icg_scale = 1.5`
- ImageNet 512×512: `icg_scale = 2.0`
- Text-to-image models: `icg_scale = 3.0`

## Code Quality Requirements

1. **Follow existing code style**: Match the style of guided-diffusion repository
2. **Add docstrings**: Document all new functions and parameters
3. **Use type hints**: Where the existing code uses them
4. **Handle edge cases**: Check for None values, device mismatches, etc.
5. **Add comments**: Explain the ICG logic clearly
6. **Preserve compatibility**: Don't break existing functionality

## Deliverables

Please provide:

1. **Complete `scripts/icg_sample.py` file**
   - Ready to run
   - Fully commented
   - With proper imports and argparse

2. **Modified `guided_diffusion/gaussian_diffusion.py`**
   - Show only the new `p_sample_loop_icg()` method
   - Include full implementation
   - With comprehensive docstring

3. **Usage example**
   - Command to run the script
   - Expected output format
   - How to verify it's working

4. **Brief explanation** of:
   - What changes you made and why
   - How ICG differs from classifier guidance
   - Any assumptions or design decisions

## Background Information

**Why ICG Works:**
The key insight from the paper is that when you feed a random, statistically independent condition `ŷ` to a conditional model, the output approximates the unconditional score:

∇ log p(x_t | ŷ) ≈ ∇ log p(x_t)

This allows you to construct a guidance signal without training an unconditional model or a classifier.

**Comparison with Classifier-Free Guidance (CFG):**
- CFG requires training with periodic null/empty conditioning (label dropping)
- ICG works on ANY conditional model, even those trained without label dropping
- Both use the same guidance formula at inference time
- ICG is more flexible and works retroactively on existing models

**Paper References:**
- ICG Paper: "No Training, No Problem: Rethinking Classifier-Free Guidance for Diffusion Models" (Sadat et al., ICLR 2025)
- Original Repo: "Diffusion Models Beat GANs on Image Synthesis" (Dhariwal & Nichol, NeurIPS 2021)

## Success Criteria

Your implementation is successful if:

✅ Code runs without errors  
✅ Generates samples similar in quality to classifier guidance  
✅ Uses only the diffusion model (no classifier)  
✅ Completes in similar time to baseline sampling  
✅ Follows the repository's code style  
✅ Is well-documented and maintainable  

## Questions to Consider

As you implement, think about:

1. How should you handle different types of conditioning (class labels vs embeddings)?
2. What's the best way to sample random conditions for your use case?
3. Should you add validation to ensure `y` and `y_random` are truly independent?
4. How can you make the code flexible for future extensions (e.g., TSG)?
5. What error handling is needed for edge cases?

Good luck! Focus on clean, well-documented code that matches the existing repository style. The core logic is straightforward—you're just replacing classifier gradients with a second forward pass using a random condition.
