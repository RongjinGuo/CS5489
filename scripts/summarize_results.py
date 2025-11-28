"""
Summarize all experimental results for report
"""
import json
import yaml
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_training_history(model_name, checkpoint_dir="checkpoints"):
    """Load training history for a model"""
    history_file = Path(checkpoint_dir) / model_name / "history.json"
    if history_file.exists():
        with open(history_file, 'r') as f:
            return json.load(f)
    return None


def load_cv_results(checkpoint_dir="checkpoints"):
    """Load cross-validation results"""
    cv_file = Path(checkpoint_dir) / "cv_results" / "cv_summary.json"
    if cv_file.exists():
        with open(cv_file, 'r') as f:
            return json.load(f)
    return None


def create_results_table(models=['lstm', 'gru', 'transformer'], 
                        features=['bpe', 'word'],
                        checkpoint_dir="checkpoints"):
    """Create results comparison table"""
    results = []
    
    for model in models:
        for feature in features:
            model_name = f"{model}_{feature}" if feature != 'bpe' else model
            history = load_training_history(model_name, checkpoint_dir)
            
            if history:
                results.append({
                    'Model': model.upper(),
                    'Feature': feature.upper(),
                    'Best Val Loss': f"{history['best_val_loss']:.4f}",
                    'Final Train Loss': f"{history['train_losses'][-1]:.4f}" if history['train_losses'] else 'N/A',
                    'Final Val Loss': f"{history['val_losses'][-1]:.4f}" if history['val_losses'] else 'N/A',
                    'Epochs': history['num_epochs']
                })
    
    if results:
        df = pd.DataFrame(results)
        return df
    return None


def plot_model_comparison(checkpoint_dir="checkpoints", save_path="figures/model_comparison.png"):
    """Plot training curves for all models"""
    models = ['lstm', 'gru', 'transformer']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for model in models:
        history = load_training_history(model, checkpoint_dir)
        if history and history['train_losses']:
            epochs = range(1, len(history['train_losses']) + 1)
            axes[0].plot(epochs, history['train_losses'], label=f'{model.upper()}', linewidth=2)
            axes[1].plot(epochs, history['val_losses'], label=f'{model.upper()}', linewidth=2)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Training Loss', fontsize=12)
    axes[0].set_title('Training Loss Comparison', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Validation Loss', fontsize=12)
    axes[1].set_title('Validation Loss Comparison', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved model comparison to {save_path}")
    plt.close()


def create_summary_report(checkpoint_dir="checkpoints", output_file="results/summary_report.md"):
    """Create a markdown summary report"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Experimental Results Summary\n\n")
        
        # CV Results
        cv_results = load_cv_results(checkpoint_dir)
        if cv_results:
            f.write("## Cross-Validation Results\n\n")
            f.write(f"**Number of Folds:** {cv_results['n_folds']}\n\n")
            f.write(f"**Best Hyperparameters:**\n")
            f.write(f"```json\n{json.dumps(cv_results['best_hyperparams'], indent=2)}\n```\n\n")
            f.write("**Hyperparameter Performance:**\n\n")
            f.write("| Rank | Hyperparameters | Mean Val Loss | Std |\n")
            f.write("|------|----------------|---------------|-----|\n")
            for i, result in enumerate(cv_results['summary'][:5], 1):
                hp_str = str(result['hyperparams'])[:50]
                f.write(f"| {i} | {hp_str} | {result['mean_val_loss']:.4f} | {result['std_val_loss']:.4f} |\n")
            f.write("\n")
        
        # Model Results
        f.write("## Model Performance\n\n")
        results_table = create_results_table(checkpoint_dir=checkpoint_dir)
        if results_table is not None:
            f.write(results_table.to_markdown(index=False))
            f.write("\n\n")
        
        # Training Statistics
        f.write("## Training Statistics\n\n")
        models = ['lstm', 'gru', 'transformer']
        for model in models:
            history = load_training_history(model, checkpoint_dir)
            if history:
                f.write(f"### {model.upper()}\n\n")
                f.write(f"- Best Validation Loss: {history['best_val_loss']:.4f}\n")
                f.write(f"- Total Epochs: {history['num_epochs']}\n")
                if history['train_losses']:
                    f.write(f"- Final Training Loss: {history['train_losses'][-1]:.4f}\n")
                if history['val_losses']:
                    f.write(f"- Final Validation Loss: {history['val_losses'][-1]:.4f}\n")
                f.write("\n")
    
    print(f"Summary report saved to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Summarize experimental results")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                       help="Checkpoint directory")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Output directory for summaries")
    
    args = parser.parse_args()
    
    # Create results table
    print("Creating results table...")
    table = create_results_table(checkpoint_dir=args.checkpoint_dir)
    if table is not None:
        print("\nResults Table:")
        print(table.to_string(index=False))
        
        # Save as CSV
        csv_path = Path(args.output_dir) / "results_table.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(csv_path, index=False)
        print(f"\nSaved to {csv_path}")
    
    # Plot comparison
    print("\nCreating comparison plots...")
    plot_model_comparison(checkpoint_dir=args.checkpoint_dir)
    
    # Create summary report
    print("\nCreating summary report...")
    create_summary_report(checkpoint_dir=args.checkpoint_dir,
                         output_file=f"{args.output_dir}/summary_report.md")
    
    print("\n" + "="*60)
    print("Results summarization completed!")
    print("="*60)


if __name__ == "__main__":
    main()

