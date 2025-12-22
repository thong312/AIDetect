"""
🚀 AIHumanDetect - Setup & Run Guide

Complete guide to run the project in Python
"""

import os
import sys
import subprocess
from pathlib import Path

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def run_command(cmd, description=""):
    """Run shell command with error handling"""
    if description:
        print(f"▶ {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Success")
            return True
        else:
            print(f"❌ Failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


# ============================================================================
# 1. PREREQUISITES CHECK
# ============================================================================

def check_prerequisites():
    """Check if all prerequisites are installed"""
    print_section("1. CHECKING PREREQUISITES")
    
    checks = [
        ("Python", "python3 --version"),
        ("pip", "pip3 --version"),
        ("git", "git --version"),
    ]
    
    for name, cmd in checks:
        if run_command(cmd, f"Checking {name}"):
            pass
        else:
            print(f"⚠️  {name} not found or not in PATH")


# ============================================================================
# 2. ENVIRONMENT SETUP
# ============================================================================

def setup_environment():
    """Setup virtual environment"""
    print_section("2. SETTING UP ENVIRONMENT")
    
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
    else:
        print("Creating virtual environment...")
        run_command("python3 -m venv venv", "Create venv")
    
    activate_cmd = "source venv/bin/activate" if sys.platform != "win32" else "venv\\Scripts\\activate"
    print(f"\n💡 To activate virtual environment:")
    print(f"   {activate_cmd}")


# ============================================================================
# 3. INSTALL DEPENDENCIES
# ============================================================================

def install_dependencies():
    """Install required packages"""
    print_section("3. INSTALLING DEPENDENCIES")
    
    packages = {
        "Core ML": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "transformers>=4.35.0",
        ],
        "Image Processing": [
            "Pillow>=10.0.0",
        ],
        "Web Framework": [
            "flask>=2.0.0",
            "sqlalchemy>=2.0.0",
        ],
        "Enhanced Ensemble": [
            "scikit-learn>=1.3.0",
            "matplotlib>=3.8.0",
            "numpy>=1.24.0",
            "psutil>=5.9.0",
        ],
        "Optional": [
            "tqdm>=4.66.0",
            "tensorboard>=2.14.0",
        ]
    }
    
    for category, pkgs in packages.items():
        print(f"\n{category}:")
        for pkg in pkgs:
            print(f"  • {pkg}")
    
    print("\n💡 Install with:")
    print("   pip install -r requirements_enhanced.txt")
    print("   OR")
    print("   pip install torch torchvision transformers pillow flask sqlalchemy scikit-learn matplotlib numpy psutil")


# ============================================================================
# 4. PROJECT STRUCTURE
# ============================================================================

def check_project_structure():
    """Verify project folder structure"""
    print_section("4. PROJECT STRUCTURE")
    
    required_folders = [
        "models/emotion_vit_rafdb",
        "models/gender_vit_utkface",
        "models/age_vit_deploy",
        "static",
        "templates",
    ]
    
    print("Checking required folders:\n")
    for folder in required_folders:
        path = Path(folder)
        status = "✅" if path.exists() else "❌"
        print(f"{status} {folder}")
    
    print("\n\nProject structure:")
    print("""
    AIHumanDetect/
    ├── models/                          (Pre-trained models) ✅
    │   ├── age_vit_deploy/
    │   ├── emotion_vit_rafdb/
    │   └── gender_vit_utkface/
    ├── temp/                            (Auto-created)
    ├── logs/                            (Auto-created)
    ├── static/                          (Web assets)
    ├── templates/                       (HTML templates)
    ├── Core Files
    │   ├── app.py                       (Flask web app)
    │   ├── pipeline.py                  (Processing pipeline)
    │   ├── emotion_model.py             (Emotion prediction)
    │   ├── gender_model.py              (Gender prediction)
    │   ├── age_model.py                 (Age prediction)
    │   └── ... (other core files)
    └── NEW: Enhanced Ensemble (Optional) ⭐
        ├── enhanced_ensemble.py         (Improved ensemble)
        ├── monitor.py                   (Performance monitoring)
        ├── evaluate.py                  (Evaluation framework)
        └── ... (other enhancement files)
    """)


# ============================================================================
# 5. QUICK START - WEB APP
# ============================================================================

def run_web_app():
    """Instructions to run web app"""
    print_section("5. RUNNING WEB APPLICATION")
    
    print("🌐 To start the web app:\n")
    print("1. Make sure dependencies are installed:")
    print("   pip install flask sqlalchemy transformers torch torchvision")
    print("\n2. Run the app:")
    print("   python3 app.py")
    print("\n3. Open browser:")
    print("   http://localhost:5000")
    print("\n4. Features available:")
    print("   • Register new users")
    print("   • Recognize registered faces")
    print("   • Analyze emotions, gender, age")


# ============================================================================
# 6. QUICK START - ENSEMBLE MODEL
# ============================================================================

def run_ensemble_model():
    """Instructions to run ensemble model"""
    print_section("6. RUNNING ENSEMBLE MODEL")
    
    print("🤖 To use the enhanced ensemble model:\n")
    
    print("Option A: Quick test")
    print("-" * 80)
    print("""
from PIL import Image
from enhanced_ensemble import BatchEnsemblePredictor

predictor = BatchEnsemblePredictor(batch_size=8)
image = Image.open('face.jpg').convert('RGB')
result = predictor.predict(image, return_all_scores=True)

print(f"Emotion: {result['predicted_class']}")
print(f"Ensemble confidence: {result['ensemble_confidence']:.1f}%")
print(f"ViT confidence: {result['vit_confidence']:.1f}%")
print(f"TransUNet confidence: {result['transunet_confidence']:.1f}%")
    """)
    
    print("\nOption B: Run test script")
    print("-" * 80)
    print("   python3 test_ensemble.py")
    
    print("\nOption C: Use code examples")
    print("-" * 80)
    print("   python3 QUICK_START.py")


# ============================================================================
# 7. VERIFICATION
# ============================================================================

def verify_installation():
    """Verify all packages are installed"""
    print_section("7. VERIFICATION")
    
    print("Checking Python packages...\n")
    
    packages = {
        "PyTorch": "torch",
        "Transformers": "transformers",
        "Pillow": "PIL",
        "Flask": "flask",
        "scikit-learn": "sklearn",
        "matplotlib": "matplotlib",
        "numpy": "numpy",
    }
    
    for name, import_name in packages.items():
        try:
            __import__(import_name)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - Install with: pip install {name.lower()}")
    
    # Check GPU
    print("\nGPU Status:")
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU: {device} ({memory:.1f} GB)")
        else:
            print("⚠️  GPU: Not available (will use CPU)")
    except:
        print("⚠️  Could not check GPU")


# ============================================================================
# 8. TROUBLESHOOTING
# ============================================================================

def print_troubleshooting():
    """Print common issues and solutions"""
    print_section("8. TROUBLESHOOTING")
    
    issues = [
        {
            "problem": "ModuleNotFoundError: No module named 'torch'",
            "solution": "pip install torch torchvision"
        },
        {
            "problem": "CUDA out of memory",
            "solution": "Reduce batch_size or use CPU mode (device='cpu')"
        },
        {
            "problem": "Models not found",
            "solution": "Check models/ folder exists, or models auto-download on first use"
        },
        {
            "problem": "Port 5000 already in use",
            "solution": "python3 app.py --port 8000 or kill existing process"
        },
        {
            "problem": "Slow inference",
            "solution": "Use batch processing or reduce model weight (ViT > TransUNet)"
        },
    ]
    
    for i, issue in enumerate(issues, 1):
        print(f"\n{i}. Problem: {issue['problem']}")
        print(f"   Solution: {issue['solution']}")


# ============================================================================
# 9. DOCUMENTATION
# ============================================================================

def print_documentation():
    """Print documentation references"""
    print_section("9. DOCUMENTATION")
    
    docs = [
        ("SUMMARY.md", "Overview of all improvements"),
        ("ENSEMBLE_GUIDE.md", "Ensemble model guide and usage"),
        ("QUICK_START.py", "Code examples and recipes"),
        ("IMPROVEMENTS.md", "Detailed improvements explanation"),
        ("INDEX.md", "Complete index and navigation"),
        ("COMPARISON.md", "Before/after comparison"),
        ("SETUP_GUIDE.sh", "This setup guide (shell version)"),
    ]
    
    print("📚 Documentation files:\n")
    for doc, desc in docs:
        print(f"  • {doc:<30} - {desc}")
    
    print("\n💡 Start with: cat SUMMARY.md")


# ============================================================================
# 10. QUICK COMMANDS
# ============================================================================

def print_quick_commands():
    """Print quick command reference"""
    print_section("10. QUICK COMMANDS REFERENCE")
    
    commands = [
        ("Web App", "python3 app.py"),
        ("Test Emotion Model", "python3 -c \"from emotion_model import predict_emotion; print('✅ OK')\""),
        ("Test Ensemble", "python3 test_ensemble.py"),
        ("Run Examples", "python3 QUICK_START.py"),
        ("Check GPU", "python3 -c \"import torch; print(torch.cuda.is_available())\""),
        ("Install Dependencies", "pip install -r requirements_enhanced.txt"),
        ("Create Venv", "python3 -m venv venv && source venv/bin/activate"),
    ]
    
    print("Frequently used commands:\n")
    for desc, cmd in commands:
        print(f"  {desc}:")
        print(f"    {cmd}\n")


# ============================================================================
# 11. PERFORMANCE TIPS
# ============================================================================

def print_performance_tips():
    """Print performance optimization tips"""
    print_section("11. PERFORMANCE OPTIMIZATION TIPS")
    
    tips = [
        {
            "category": "GPU Optimization",
            "tips": [
                "Check GPU with: torch.cuda.is_available()",
                "Use GPU memory efficiently with batch processing",
                "Monitor memory with torch.cuda.memory_allocated()",
            ]
        },
        {
            "category": "Batch Size Tuning",
            "tips": [
                "Large GPU (8GB+): batch_size=32",
                "Medium GPU (4GB): batch_size=16",
                "Small GPU (2GB): batch_size=4",
                "CPU only: batch_size=1",
            ]
        },
        {
            "category": "Ensemble Method Selection",
            "tips": [
                "weighted_average: fastest, default",
                "harmonic_mean: balanced",
                "product: more selective",
                "max_voting: most selective",
            ]
        },
        {
            "category": "Model Caching",
            "tips": [
                "Models are lazy loaded (on first use)",
                "Saves startup time and memory",
                "Unload with: model.unload_models()",
            ]
        },
    ]
    
    for tip_group in tips:
        print(f"\n{tip_group['category']}:")
        for tip in tip_group['tips']:
            print(f"  • {tip}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all setup checks and guides"""
    
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "🚀 AIHumanDetect - Setup & Run Guide" + " "*22 + "║")
    print("╚" + "="*78 + "╝")
    
    # Run all checks and guides
    check_prerequisites()
    setup_environment()
    install_dependencies()
    check_project_structure()
    run_web_app()
    run_ensemble_model()
    verify_installation()
    print_troubleshooting()
    print_documentation()
    print_quick_commands()
    print_performance_tips()
    
    # Final summary
    print_section("✅ SETUP COMPLETE!")
    print("""
Ready to run the project!

Next steps:

1. Install dependencies:
   pip install -r requirements_enhanced.txt

2. Start web app:
   python3 app.py

3. Or run ensemble model:
   python3 test_ensemble.py

4. Read documentation:
   cat SUMMARY.md

For questions, check INDEX.md for complete navigation.
    """)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        sys.exit(1)
