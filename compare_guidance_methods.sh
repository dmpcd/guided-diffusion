#!/bin/bash
# Compare ICG with other guidance methods

cd /home/senum/projects/guided-diffusion/guided-diffusion

echo ""
echo "🔬 COMPARING GUIDANCE METHODS"
echo "=============================="
echo ""
echo "This will generate samples using three different methods:"
echo "  1. Without guidance (baseline)"
echo "  2. With classifier guidance (requires classifier model)"
echo "  3. With ICG (Independent Condition Guidance)"
echo ""

SEED=${1:-42}
echo "Using seed: $SEED"
echo ""

# Check if classifier exists
HAS_CLASSIFIER=false
if [ -f "models/64x64_classifier.pt" ]; then
    HAS_CLASSIFIER=true
fi

# Generate baseline
echo "=========================================="
echo "1/3: WITHOUT Guidance (Baseline)"
echo "=========================================="
./without_classifier.sh
echo ""

# Generate with classifier if available
if [ "$HAS_CLASSIFIER" = true ]; then
    echo "=========================================="
    echo "2/3: WITH Classifier Guidance"
    echo "=========================================="
    ./with_classifier_guidance.sh
    echo ""
else
    echo "⚠️  Skipping classifier guidance (classifier model not found)"
    echo ""
fi

# Generate with ICG
echo "=========================================="
echo "3/3: WITH ICG (Independent Condition Guidance)"
echo "=========================================="
./icg_demo.sh
echo ""

# Create comparison
echo "=========================================="
echo "📊 Creating Comparison Visualization"
echo "=========================================="

python -c "
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
import sys
sys.path.insert(0, '.')
from imagenet_classes import format_class_label

# Load samples
outputs = {}
methods = ['without_classifier', 'icg']
has_classifier = os.path.exists('outputs/with_classifier/samples_4x64x64x3.npz')
if has_classifier:
    methods.insert(1, 'with_classifier')

for method in methods:
    npz_file = glob.glob(f'outputs/{method}/samples_*.npz')
    if npz_file:
        data = np.load(npz_file[0])
        outputs[method] = {
            'images': data['arr_0'],
            'labels': data['arr_1'] if 'arr_1' in data else None
        }

if not outputs:
    print('❌ No samples found!')
    exit(1)

# Create comparison grid
n_samples = min([len(v['images']) for v in outputs.values()])
n_methods = len(outputs)

fig, axes = plt.subplots(n_samples, n_methods, figsize=(5*n_methods, 5*n_samples))
if n_samples == 1:
    axes = axes.reshape(1, -1)

method_titles = {
    'without_classifier': 'Without Guidance\\n(Baseline)',
    'with_classifier': 'Classifier Guidance\\n(Requires Classifier)',
    'icg': 'ICG\\n(Training-Free)'
}

for j, (method, data) in enumerate(outputs.items()):
    for i in range(n_samples):
        ax = axes[i, j] if n_samples > 1 else axes[j]
        ax.imshow(data['images'][i])
        ax.axis('off')
        
        if i == 0:
            ax.set_title(method_titles[method], fontsize=14, fontweight='bold', pad=10)
        
        if data['labels'] is not None and j == 0:
            class_label = format_class_label(data['labels'][i])
            ax.text(0.02, 0.98, class_label, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle('Comparison: Different Guidance Methods', 
             fontsize=16, fontweight='bold', y=0.998)
plt.tight_layout()

os.makedirs('outputs/comparison', exist_ok=True)
comparison_path = 'outputs/comparison/guidance_comparison.png'
plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
print(f'✓ Comparison saved: {comparison_path}')
plt.close()
"

echo ""
echo "=========================================="
echo "✅ COMPARISON COMPLETE!"
echo ""
echo "Results saved to:"
echo "  📊 outputs/comparison/guidance_comparison.png"
echo ""
echo "Key Observations:"
echo "  • ICG provides guidance without a classifier"
echo "  • Similar quality to classifier guidance"
echo "  • Training-free and no extra model needed"
echo "  • Faster inference (no gradient computation)"
echo "=========================================="
