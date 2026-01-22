"""
Analyze and visualize experimental results.
"""

import argparse
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_results(filepath):
    """Load results JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_crc_history(filepath):
    """Load CRC history pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def plot_crc_training_dynamics(crc_history, save_path=None):
    """
    Plot CRC training dynamics: q, mu, risk, coverage over time.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot q threshold
    if 'q' in crc_history and len(crc_history['q']) > 0:
        axes[0, 0].plot(crc_history['q'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Calibration Step')
        axes[0, 0].set_ylabel('CRC Threshold q')
        axes[0, 0].set_title('CRC Threshold Evolution')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot mu penalty weight
    if 'mu' in crc_history and len(crc_history['mu']) > 0:
        axes[0, 1].plot(crc_history['mu'], 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Calibration Step')
        axes[0, 1].set_ylabel('Penalty Weight μ')
        axes[0, 1].set_title('Dual Variable μ Evolution')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot calibration risk
    if 'cal_risk' in crc_history and len(crc_history['cal_risk']) > 0:
        axes[1, 0].plot(crc_history['cal_risk'], 'g-', linewidth=2, label='Cal Risk')
        if 'alpha' in crc_history:
            axes[1, 0].axhline(y=crc_history['alpha'], color='r', linestyle='--', label='Target α')
        axes[1, 0].set_xlabel('Calibration Step')
        axes[1, 0].set_ylabel('Risk')
        axes[1, 0].set_title('Calibration Set Risk')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot calibration coverage
    if 'cal_coverage' in crc_history and len(crc_history['cal_coverage']) > 0:
        axes[1, 1].plot(crc_history['cal_coverage'], 'purple', linewidth=2)
        axes[1, 1].set_xlabel('Calibration Step')
        axes[1, 1].set_ylabel('Coverage')
        axes[1, 1].set_title('Calibration Set Coverage')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training dynamics to {save_path}")
    else:
        plt.show()
    
    plt.close()


def compare_methods_bar(results_dict, save_path=None):
    """
    Bar chart comparing all methods on key metrics.
    """
    methods = list(results_dict.keys())
    
    # Extract metrics
    coverages = []
    risks = []
    dars = []
    
    for method in methods:
        r = results_dict[method]
        if 'coverage' in r:
            if isinstance(r['coverage'], dict) and 'mean' in r['coverage']:
                coverages.append(r['coverage']['mean'])
                risks.append(r['risk']['mean'])
                dars.append(r['dar']['mean'])
            else:
                coverages.append(r.get('coverage', 0))
                risks.append(r.get('risk', 0))
                dars.append(r.get('dar', 0))
    
    x = np.arange(len(methods))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(x - width, coverages, width, label='Coverage (higher better)', color='blue', alpha=0.7)
    ax.bar(x, [1-r for r in risks], width, label='1 - Risk (higher better)', color='green', alpha=0.7)
    ax.bar(x + width, [1-d for d in dars], width, label='1 - DAR (higher better)', color='red', alpha=0.7)
    
    ax.set_ylabel('Score')
    ax.set_title('Method Comparison (All metrics normalized: higher is better)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


def violation_rate_analysis(all_results, alpha, save_path=None):
    """
    Analyze violation rates across seeds for each method.
    """
    methods = {}
    
    for result in all_results:
        for method, data in result.items():
            if method not in methods:
                methods[method] = []
            
            # Extract risk
            if 'test_risk' in data:
                risk = data['test_risk']
            elif 'risk' in data:
                if isinstance(data['risk'], dict):
                    risk = data['risk']['mean']
                else:
                    risk = data['risk']
            else:
                continue
            
            methods[method].append(risk)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method, risks in methods.items():
        risks = np.array(risks)
        violations = risks > alpha
        violation_rate = np.mean(violations)
        
        # Histogram
        ax.hist(risks, alpha=0.5, label=f'{method} (viol: {violation_rate:.1%})', bins=10)
    
    ax.axvline(x=alpha, color='r', linestyle='--', linewidth=2, label=f'Target α = {alpha}')
    ax.set_xlabel('Risk')
    ax.set_ylabel('Frequency')
    ax.set_title('Risk Distribution Across Seeds (Violation Rate Analysis)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved violation analysis to {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_summary_table(results):
    """Print a nice summary table."""
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}\n")
    
    header = f"{'Method':<20} {'Coverage':<12} {'Risk':<12} {'Accuracy':<12} {'DAR':<12}"
    print(header)
    print("-" * 80)
    
    for method, data in results.items():
        if 'coverage' in data:
            if isinstance(data['coverage'], dict):
                cov = f"{data['coverage']['mean']:.3f}±{data['coverage']['std']:.3f}"
                risk = f"{data['risk']['mean']:.4f}±{data['risk']['std']:.4f}"
                acc = f"{data['accuracy']['mean']:.3f}±{data['accuracy']['std']:.3f}"
                dar = f"{data['dar']['mean']:.3f}±{data['dar']['std']:.3f}"
            else:
                cov = f"{data.get('coverage', 0):.3f}"
                risk = f"{data.get('risk', 0):.4f}"
                acc = f"{data.get('accuracy', 0):.3f}"
                dar = f"{data.get('dar', 0):.3f}"
            
            print(f"{method:<20} {cov:<12} {risk:<12} {acc:<12} {dar:<12}")
    
    print(f"\n{'='*80}\n")


def main(exp_name='comparison', output_dir='results/analysis'):
    """
    Main analysis function.
    """
    print(f"\n{'='*70}")
    print("ANALYZING EXPERIMENTAL RESULTS")
    print(f"{'='*70}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load aggregated results
    agg_file = f'results/{exp_name}_aggregated.json'
    if os.path.exists(agg_file):
        print(f"Loading aggregated results: {agg_file}")
        aggregated = load_results(agg_file)
        
        # Print summary
        print_summary_table(aggregated)
        
        # Generate comparison plots
        compare_methods_bar(aggregated, save_path=f'{output_dir}/methods_comparison.png')
    else:
        print(f"Aggregated results not found: {agg_file}")
    
    # Look for CRC-Select training history
    crc_history_files = [f for f in os.listdir('checkpoints') if 'crc_history' in f]
    if crc_history_files:
        print(f"\nFound {len(crc_history_files)} CRC history files")
        for hist_file in crc_history_files[:3]:  # Plot first 3
            print(f"Analyzing: {hist_file}")
            hist_path = os.path.join('checkpoints', hist_file)
            crc_hist = load_crc_history(hist_path)
            
            save_name = hist_file.replace('.pkl', '.png')
            plot_crc_training_dynamics(
                crc_hist,
                save_path=f'{output_dir}/{save_name}'
            )
    
    # Load individual seed results for violation analysis
    seed_files = [f for f in os.listdir('results') if f.startswith(f'{exp_name}_seed') and f.endswith('_all.json')]
    if seed_files:
        print(f"\nFound {len(seed_files)} seed result files")
        all_results = []
        for seed_file in seed_files:
            results = load_results(f'results/{seed_file}')
            all_results.append(results)
        
        # Violation rate analysis
        violation_rate_analysis(
            all_results,
            alpha=0.05,
            save_path=f'{output_dir}/violation_analysis.png'
        )
    
    print(f"\n{'='*70}")
    print(f"Analysis complete! Results saved to {output_dir}/")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze Experimental Results')
    parser.add_argument('--exp_name', type=str, default='comparison',
                       help='Experiment name')
    parser.add_argument('--output_dir', type=str, default='results/analysis',
                       help='Output directory for analysis')
    
    args = parser.parse_args()
    
    main(exp_name=args.exp_name, output_dir=args.output_dir)

