#!/usr/bin/env python3
"""
Fix Python Path for Try4 Scripts

This script needs to be run before main_pipeline.py to ensure all
subdirectory scripts can import from config.config properly.

Creates __init__.py files in all Try4 subdirectories.
"""

from pathlib import Path

def fix_try4_imports():
    """Create __init__.py files in all Try4 subdirectories"""
    
    try4_dir = Path(__file__).parent
    
    subdirs = [
        'config',
        'feature_extraction',
        'preprocessing',
        'modeling'
    ]
    
    print("🔧 Fixing Try4 Python imports...")
    print("="*60)
    
    for subdir in subdirs:
        init_file = try4_dir / subdir / '__init__.py'
        
        if not init_file.exists():
            init_file.write_text(f"# {subdir.replace('_', ' ').title()} package\n")
            print(f"✅ Created: {subdir}/__init__.py")
        else:
            print(f"✓ Exists: {subdir}/__init__.py")
    
    print("="*60)
    print("✅ Import structure fixed!")
    print("\nYou can now run:")
    print("  python main_pipeline.py")
    print()

if __name__ == '__main__':
    fix_try4_imports()
