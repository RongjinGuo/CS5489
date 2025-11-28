#!/usr/bin/env python3
"""
Complete pipeline script to run all steps of the MT project
"""
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error in {description}:")
        print(result.stderr)
        return False
    else:
        print(result.stdout)
        return True

def main():
    """Run complete pipeline"""
    steps = [
        # ("python data/download_data.py", "Download IWSLT17 dataset"),
        ("python data/prepare_iwslt17.py", "Prepare and split dataset"),
        ("python src/data_exploration.py", "Data exploration and visualization"),
    ]
    
    models = ["lstm", "gru", "transformer"]
    
    print("="*60)
    print("Machine Translation Project - Complete Pipeline")
    print("="*60)
    
    # Step 1-3: Data preparation
    for cmd, desc in steps:
        if not run_command(cmd, desc):
            print(f"Failed at: {desc}")
            sys.exit(1)
    
    # Step 4: Train models
    print("\n" + "="*60)
    print("Training Models")
    print("="*60)
    
    for model in models:
        cmd = f"python scripts/train.py --model {model} --config config.yaml"
        if not run_command(cmd, f"Train {model.upper()} model"):
            print(f"Warning: Training {model} failed, continuing...")
    
    # Step 5: Evaluate models
    print("\n" + "="*60)
    print("Evaluating Models")
    print("="*60)
    
    for model in models:
        checkpoint = f"checkpoints/{model}_best.pt"
        if Path(checkpoint).exists():
            cmd = f"python scripts/evaluate.py --model {model} --checkpoint {checkpoint} --config config.yaml"
            run_command(cmd, f"Evaluate {model.upper()} model")
        else:
            print(f"Warning: Checkpoint {checkpoint} not found, skipping evaluation")
    
    # Step 6: Visualizations
    print("\n" + "="*60)
    print("Generating Visualizations")
    print("="*60)
    
    for model in models:
        checkpoint = f"checkpoints/{model}_best.pt"
        if Path(checkpoint).exists():
            cmd = f"python scripts/visualize.py --model {model} --checkpoint {checkpoint} --config config.yaml --task all"
            run_command(cmd, f"Visualize {model.upper()} model")
        else:
            print(f"Warning: Checkpoint {checkpoint} not found, skipping visualization")
    
    print("\n" + "="*60)
    print("Pipeline completed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Check results/ directory for BLEU scores and examples")
    print("2. Check figures/ directory for visualizations")
    print("3. Review report/report_template.md and fill in your results")

if __name__ == "__main__":
    main()

