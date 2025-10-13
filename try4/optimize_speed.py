#!/usr/bin/env python3
"""
Auto-detect SageMaker instance and optimize Try4 config for maximum speed
"""

import subprocess
import re
import sys
from pathlib import Path

def get_instance_type():
    """Detect SageMaker instance type from system info"""
    try:
        # Try to get from EC2 metadata
        result = subprocess.run(
            ['curl', '-s', 'http://169.254.169.254/latest/meta-data/instance-type'],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout.strip()
    except:
        pass
    
    # Fallback: Detect from CPU/RAM
    try:
        # Get CPU count
        with open('/proc/cpuinfo') as f:
            cpu_count = len([l for l in f if l.startswith('processor')])
        
        # Get RAM in GB
        with open('/proc/meminfo') as f:
            mem_kb = int(f.readline().split()[1])
            ram_gb = mem_kb // (1024 * 1024)
        
        # Map to likely instance
        if cpu_count == 4 and ram_gb <= 20:
            return 'ml.g5.xlarge'
        elif cpu_count == 8 and ram_gb <= 40:
            return 'ml.g5.2xlarge'
        elif cpu_count == 16 and ram_gb <= 80:
            return 'ml.g5.4xlarge'
        elif cpu_count >= 48:
            return 'ml.g5.12xlarge'
    except:
        pass
    
    return 'unknown'

def get_optimal_config(instance_type):
    """Get optimal batch sizes and workers for instance type"""
    configs = {
        'ml.g5.xlarge': {
            'text_batch': 24,
            'image_batch': 96,
            'workers': 4,
            'expected_time': '2-2.5 hours'
        },
        'ml.g5.2xlarge': {
            'text_batch': 32,
            'image_batch': 128,
            'workers': 6,
            'expected_time': '50-70 minutes ✅ UNDER 1 HOUR'
        },
        'ml.g5.4xlarge': {
            'text_batch': 48,
            'image_batch': 192,
            'workers': 12,
            'expected_time': '35-50 minutes ✅✅ FAST'
        },
        'ml.g5.12xlarge': {
            'text_batch': 64,
            'image_batch': 256,
            'workers': 24,
            'expected_time': '20-30 minutes ✅✅✅ EXTREME'
        }
    }
    
    return configs.get(instance_type, configs['ml.g5.xlarge'])

def optimize_config(config_path, instance_type):
    """Optimize config.py for the detected instance"""
    
    # Read config
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Get optimal settings
    opt = get_optimal_config(instance_type)
    
    # Apply optimizations
    # Text model batch size
    content = re.sub(
        r"('batch_size':\s*)\d+",
        f"\\g<1>{opt['text_batch']}",
        content,
        count=1
    )
    
    # Image model batch size (second occurrence)
    lines = content.split('\n')
    batch_count = 0
    for i, line in enumerate(lines):
        if "'batch_size':" in line and 'TEXT_MODEL' not in ''.join(lines[max(0,i-10):i]):
            if batch_count == 1:  # IMAGE_MODEL
                lines[i] = re.sub(r"'batch_size':\s*\d+", f"'batch_size': {opt['image_batch']}", line)
            batch_count += 1
    content = '\n'.join(lines)
    
    # Workers
    content = re.sub(
        r"('num_workers':\s*)\d+",
        f"\\g<1>{opt['workers']}",
        content
    )
    
    # Write optimized config
    with open(config_path, 'w') as f:
        f.write(content)
    
    return opt

def main():
    print("="*80)
    print("🚀 TRY4 SPEED OPTIMIZER")
    print("="*80)
    print()
    
    # Detect instance
    instance_type = get_instance_type()
    print(f"📍 Detected instance: {instance_type}")
    print()
    
    # Get config path
    script_dir = Path(__file__).parent
    config_path = script_dir / 'config' / 'config.py'
    
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        sys.exit(1)
    
    # Optimize
    print("🔧 Optimizing configuration...")
    opt = optimize_config(config_path, instance_type)
    print()
    
    # Show results
    print("="*80)
    print("✅ OPTIMIZATION COMPLETE")
    print("="*80)
    print()
    print(f"📊 Optimized for: {instance_type}")
    print(f"   Text batch size:  {opt['text_batch']}")
    print(f"   Image batch size: {opt['image_batch']}")
    print(f"   Data workers:     {opt['workers']}")
    print()
    print(f"⏱️  Expected pipeline time: {opt['expected_time']}")
    print()
    print("="*80)
    print()
    print("🚀 Ready to run:")
    print("   python main_pipeline.py")
    print()

if __name__ == '__main__':
    main()
