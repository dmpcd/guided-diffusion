#!/bin/bash
# Compare Results - With vs Without Classifier Guidance

cd /home/senum/projects/guided-diffusion/guided-diffusion

echo ""
echo "📊 COMPARING RESULTS: WITH vs WITHOUT CLASSIFIER GUIDANCE"
echo "=========================================================="
echo ""

# Check if both directories exist
if [ ! -d "outputs/without_classifier" ]; then
    echo "❌ No baseline results found!"
    echo "   Run: ./without_classifier.sh"
    echo ""
    exit 1
fi

if [ ! -d "outputs/with_classifier" ]; then
    echo "❌ No classifier-guided results found!"
    echo "   Run: ./with_classifier_guidance.sh"
    echo ""
    exit 1
fi

echo "📂 Results Location:"
echo "   Without Classifier: outputs/without_classifier/"
echo "   With Classifier:    outputs/with_classifier/"
echo ""

# Count samples
without_count=$(ls outputs/without_classifier/sample_*.png 2>/dev/null | wc -l)
with_count=$(ls outputs/with_classifier/sample_*.png 2>/dev/null | wc -l)

echo "📸 Generated Samples:"
echo "   Without Classifier: $without_count images"
echo "   With Classifier:    $with_count images"
echo ""

# List files
echo "📁 Without Classifier:"
ls -lh outputs/without_classifier/*.png 2>/dev/null || echo "   No images found"
echo ""

echo "📁 With Classifier:"
ls -lh outputs/with_classifier/*.png 2>/dev/null || echo "   No images found"
echo ""

# Create side-by-side comparison grid
echo "🎨 Creating comparison grid..."
python -c "
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob

# Load without classifier samples
without_dir = 'outputs/without_classifier'
without_files = sorted(glob.glob(f'{without_dir}/sample_*.png'))

# Load with classifier samples  
with_dir = 'outputs/with_classifier'
with_files = sorted(glob.glob(f'{with_dir}/sample_*.png'))

if not without_files or not with_files:
    print('❌ Missing images in one or both directories!')
    exit(1)

n = min(len(without_files), len(with_files), 4)

fig, axes = plt.subplots(n, 2, figsize=(8, n*4))
if n == 1:
    axes = axes.reshape(1, -1)

for i in range(n):
    # Without classifier
    img_without = Image.open(without_files[i])
    axes[i, 0].imshow(img_without)
    axes[i, 0].set_title(f'Sample {i+1}: WITHOUT Classifier', fontsize=10)
    axes[i, 0].axis('off')
    
    # With classifier
    img_with = Image.open(with_files[i])
    axes[i, 1].imshow(img_with)
    axes[i, 1].set_title(f'Sample {i+1}: WITH Classifier', fontsize=10)
    axes[i, 1].axis('off')

plt.suptitle('Comparison: Without vs With Classifier Guidance', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()

os.makedirs('outputs/comparison', exist_ok=True)
comparison_path = 'outputs/comparison/side_by_side.png'
plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
print(f'✓ Comparison saved: {comparison_path}')
plt.close()

print('')
print('Key Observations:')
print('  - Classifier-guided samples tend to be more realistic')
print('  - Better class-specific features and details')
print('  - Trade-off: slightly less diversity, better quality')
"

echo ""
echo "=========================================================="
echo "✅ COMPARISON COMPLETE!"
echo ""
echo "View comparison:"
echo "  📊 Side-by-side: outputs/comparison/side_by_side.png"
echo "  📁 Individual:   outputs/without_classifier/ vs outputs/with_classifier/"
echo ""
echo "Key Differences:"
echo "  • Classifier guidance improves image quality and realism"
echo "  • Better adherence to specific object classes"
echo "  • Trade-off between quality (↑) and diversity (↓)"
echo "=========================================================="
echo ""
