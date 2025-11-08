#!/bin/bash
# Calculate FID score comparing with/without classifier guidance

cd /home/senum/projects/guided-diffusion/guided-diffusion

echo "📊 FID SCORE EVALUATION"
echo "======================="
echo ""

# Check if evaluation dependencies are installed
python -c "import tensorflow" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  TensorFlow not installed. Installing evaluation dependencies..."
    pip install -r evaluations/requirements.txt
    echo ""
fi

# Check if we have FID evaluation samples, otherwise use demo samples
if [ -d "outputs/fid_evaluation/without_classifier" ] && [ -n "$(ls outputs/fid_evaluation/without_classifier/samples_*.npz 2>/dev/null)" ]; then
    WITHOUT_SAMPLES=$(ls outputs/fid_evaluation/without_classifier/samples_*.npz | head -1)
    WITH_SAMPLES=$(ls outputs/fid_evaluation/with_classifier/samples_*.npz | head -1)
    echo "Using FID evaluation samples"
elif [ -f "outputs/without_classifier/samples_4x64x64x3.npz" ]; then
    WITHOUT_SAMPLES="outputs/without_classifier/samples_4x64x64x3.npz"
    WITH_SAMPLES="outputs/with_classifier/samples_4x64x64x3.npz"
    echo "⚠️  Using demo samples (only 4 images - not reliable for FID!)"
    echo "   For proper evaluation, run: ./generate_for_fid.sh 100"
    echo ""
else
    echo "❌ No samples found!"
    echo "   Run: ./generate_for_fid.sh 100"
    exit 1
fi

echo "📥 Downloading ImageNet 64x64 reference batch..."
echo "   (This is ~340MB, one-time download)"
echo ""

mkdir -p reference_batches
cd reference_batches

if [ ! -f "VIRTUAL_imagenet64_labeled.npz" ]; then
    wget -q --show-progress https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/64/VIRTUAL_imagenet64_labeled.npz
    echo ""
fi

cd ..

echo "======================="
echo "📊 Evaluating WITHOUT Classifier"
echo "======================="
python evaluations/evaluator.py \
    reference_batches/VIRTUAL_imagenet64_labeled.npz \
    "$WITHOUT_SAMPLES"

echo ""
echo "======================="
echo "📊 Evaluating WITH Classifier"
echo "======================="
python evaluations/evaluator.py \
    reference_batches/VIRTUAL_imagenet64_labeled.npz \
    "$WITH_SAMPLES"

echo ""
echo "======================="
echo "✅ FID Evaluation Complete!"
echo ""
echo "Note: With only 4 samples, the scores won't be very meaningful."
echo "For reliable FID scores, you need at least ~2,000 samples."
echo ""
echo "To generate more samples, increase --num_samples in the scripts."
echo "======================="
