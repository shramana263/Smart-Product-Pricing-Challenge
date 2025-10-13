# 🔧 CLIP Installation Guide for Try4

## ⚠️ Important: CLIP Dependency Fix

The `clip-by-openai` package has a dependency conflict with PyTorch 2.0+. We need to install CLIP from the official OpenAI repository instead.

---

## 📦 Installation Steps

### **Step 1: Install Main Requirements**
```bash
pip install -r requirements.txt
```

### **Step 2: Install CLIP from Official Repo**
```bash
pip install git+https://github.com/openai/CLIP.git
```

**Alternative (if git not available):**
```bash
pip install clip-openai
```

---

## ✅ Verify Installation

```bash
python -c "import torch; import clip; print('✅ CLIP installed successfully')"
```

**Expected output:**
```
✅ CLIP installed successfully
```

---

## 🎯 Complete Installation Workflow

### **On SageMaker Code Editor:**

```bash
# Navigate to try4 directory
cd ~/Smart-Product-Pricing-Challenge/try4

# Create/activate conda environment
conda create -n try4 python=3.10 -y
conda activate try4

# Install main requirements
pip install -r requirements.txt

# Install CLIP from official repo
pip install git+https://github.com/openai/CLIP.git

# Verify system
python check_system.py

# Run pipeline
python main_pipeline.py
```

### **On Local Machine with GPU:**

```bash
cd try4

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Install CLIP
pip install git+https://github.com/openai/CLIP.git

# Verify
python check_system.py
```

---

## 🔍 What's the Difference?

| Package | PyTorch Version | Status |
|---------|-----------------|--------|
| `clip-by-openai` | 1.7.1 (OLD) | ❌ Conflicts with Try4 |
| Official CLIP repo | 2.0+ (NEW) | ✅ Compatible |

---

## 🐛 Troubleshooting

### **Error: "Cannot install clip-by-openai and torch>=2.0.0"**

**Solution:** Remove `clip-by-openai` from requirements.txt and install from GitHub:
```bash
pip uninstall clip-by-openai -y
pip install git+https://github.com/openai/CLIP.git
```

### **Error: "No module named 'clip'"**

**Solution:** Install CLIP after PyTorch:
```bash
pip install torch torchvision
pip install git+https://github.com/openai/CLIP.git
```

### **Error: "git command not found"**

**Option 1 - Install git:**
```bash
# On SageMaker/Linux
sudo yum install git -y

# On Ubuntu
sudo apt-get install git -y
```

**Option 2 - Use alternative package:**
```bash
pip install clip-openai
```

---

## 📋 Updated Requirements Summary

**After removing `clip-by-openai`, install separately:**

```bash
# requirements.txt (updated)
torch>=2.0.0
transformers>=4.35.0
# ... other packages ...
# CLIP removed from here

# Install CLIP separately
pip install git+https://github.com/openai/CLIP.git
```

---

## ✅ Verification Script

Save as `verify_clip.py`:

```python
#!/usr/bin/env python3
"""Verify CLIP installation"""

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    
    import clip
    print(f"✅ CLIP: Installed")
    
    # Test loading model
    model, preprocess = clip.load("ViT-L/14", device="cpu")
    print(f"✅ CLIP ViT-L/14: Loaded successfully")
    
    print("\n🎉 All checks passed! CLIP is ready to use.")
    
except ImportError as e:
    print(f"❌ Error: {e}")
    print("\n💡 Fix: pip install git+https://github.com/openai/CLIP.git")
except Exception as e:
    print(f"⚠️ Warning: {e}")
```

**Run:**
```bash
python verify_clip.py
```

---

## 🚀 Ready to Go!

Once CLIP is installed:
```bash
python check_system.py  # Verify all prerequisites
python main_pipeline.py  # Run full pipeline
```

---

## 📚 Additional Resources

- **Official CLIP Repo:** https://github.com/openai/CLIP
- **CLIP Paper:** https://arxiv.org/abs/2103.00020
- **PyTorch Compatibility:** CLIP works with PyTorch 1.7.1+ including 2.0+

---

**Summary:** Use `pip install git+https://github.com/openai/CLIP.git` instead of `clip-by-openai` to avoid dependency conflicts! ✅
