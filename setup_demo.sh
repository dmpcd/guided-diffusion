#!/bin/bash
# Pre-Demo Setup Script
# Run this BEFORE your presentation to prepare everything

cd /home/senum/projects/guided-diffusion/guided-diffusion

echo "🔧 PRE-DEMO SETUP"
echo "================="
echo ""

# Check Python packages
echo "1️⃣ Checking Python environment..."
python -c "import torch; import guided_diffusion; print('   ✓ Core packages OK')" 2>/dev/null || echo "   ❌ Installation issue!"

# Check GPU
echo "2️⃣ Checking GPU..."
python -c "import torch; print(f'   ✓ CUDA: {torch.cuda.is_available()}'); print(f'   ✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" 2>/dev/null

# Check disk space
echo "3️⃣ Checking disk space..."
df -h . | tail -1 | awk '{print "   Available: " $4}'

# Create directories
echo "4️⃣ Creating directories..."
mkdir -p models demo_output
echo "   ✓ models/ and demo_output/ created"

# Check internet connection
echo "5️⃣ Checking internet..."
if ping -c 1 openaipublic.blob.core.windows.net &> /dev/null; then
    echo "   ✓ Internet connection OK"
else
    echo "   ⚠️  Cannot reach model server (might be slow)"
fi

# Download model (THIS IS THE KEY STEP!)
echo ""
echo "6️⃣ Downloading model (this will take 2-3 minutes)..."
if [ ! -f "models/64x64_diffusion.pt" ]; then
    wget -q --show-progress \
        https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_diffusion.pt \
        -P models/
    
    if [ -f "models/64x64_diffusion.pt" ]; then
        echo "   ✓ Model downloaded successfully!"
        ls -lh models/64x64_diffusion.pt
    else
        echo "   ❌ Download failed!"
        exit 1
    fi
else
    echo "   ✓ Model already downloaded"
    ls -lh models/64x64_diffusion.pt
fi

# Make scripts executable
echo ""
echo "7️⃣ Setting up demo scripts..."
chmod +x quick_demo.sh 2>/dev/null
echo "   ✓ Scripts ready"

echo ""
echo "=" * 50
echo "✅ SETUP COMPLETE!"
echo ""
echo "📋 You are ready to present!"
echo ""
echo "🎬 To run the demo, use:"
echo "   ./quick_demo.sh"
echo ""
echo "📖 For full instructions, see:"
echo "   DEMO_PRESENTATION.md"
echo ""
