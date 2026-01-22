"""
Evaluation utilities for CRC-Select experiments
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from typing import Dict, List, Tuple

from crc_utils import (
    compute_risk_scores,
    compute_coverage_at_risk,
    compute_ood_metrics,
    evaluate_crc
)


def plot_rc_curve(results_dict: Dict[str, List[Tuple[float, float]]],
                 save_path: str = None,
                 title: str = "Risk-Coverage Curve"):
    """
    Plot Risk-Coverage curve for multiple methods.
    
    Args:
        results_dict: {method_name: [(coverage, risk), ...]}
        save_path: path to save figure
        title: plot title
    """
    plt.figure(figsize=(8, 6))
    
    for method_name, points in results_dict.items():
        coverages = [p[0] for p in points]
        risks = [p[1] for p in points]
        plt.plot(coverages, risks, marker='o', label=method_name, linewidth=2)
    
    plt.xlabel('Coverage', fontsize=12)
    plt.ylabel('Risk', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved RC curve to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_ood_comparison(ood_results: Dict[str, Dict],
                       save_path: str = None,
                       title: str = "OOD Dangerous Acceptance Rate"):
    """
    Plot OOD metrics comparison.
    
    Args:
        ood_results: {method_name: {'dar': ..., 'id_coverage': ..., ...}}
        save_path: path to save
        title: title
    """
    methods = list(ood_results.keys())
    dar_values = [ood_results[m]['dar'] for m in methods]
    id_cov_values = [ood_results[m]['id_coverage'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, dar_values, width, label='DAR (lower is better)', color='red', alpha=0.7)
    ax.bar(x + width/2, id_cov_values, width, label='ID Coverage (higher is better)', color='blue', alpha=0.7)
    
    ax.set_ylabel('Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved OOD comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_results_table(results: Dict[str, Dict],
                        save_path: str = None) -> str:
    """
    Create formatted results table.
    
    Args:
        results: {method_name: {metric_name: value}}
        save_path: path to save table (txt/csv)
    
    Returns:
        formatted string
    """
    # Find all metrics
    all_metrics = set()
    for method_results in results.values():
        all_metrics.update(method_results.keys())
    all_metrics = sorted(list(all_metrics))
    
    # Build table
    lines = []
    header = "Method".ljust(25) + "".join([m.ljust(15) for m in all_metrics])
    lines.append(header)
    lines.append("-" * len(header))
    
    for method_name, metrics in results.items():
        row = method_name.ljust(25)
        for metric in all_metrics:
            value = metrics.get(metric, np.nan)
            if isinstance(value, float):
                row += f"{value:.4f}".ljust(15)
            else:
                row += str(value).ljust(15)
        lines.append(row)
    
    table_str = "\n".join(lines)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(table_str)
        print(f"Saved results table to {save_path}")
    
    return table_str


def evaluate_model_full(model,
                       x_test,
                       y_test,
                       x_ood=None,
                       alpha_values=[0.01, 0.05, 0.10],
                       loss_fn='cross_entropy',
                       ood_name='SVHN') -> Dict:
    """
    Full evaluation of a model.
    
    Args:
        model: trained model
        x_test: ID test images
        y_test: ID test labels
        x_ood: OOD images (optional)
        alpha_values: list of target risk levels
        loss_fn: loss function
        ood_name: name of OOD dataset
    
    Returns:
        results dict with all metrics
    """
    results = {}
    
    # Predict on ID test
    predictions, pred_aux = model.predict(x_test, batch_size=128)
    pred_probs = predictions[:, :-1]
    pred_probs = pred_probs / (np.sum(pred_probs, axis=1, keepdims=True) + 1e-7)
    selection_scores = predictions[:, -1]
    
    # ID metrics
    results['id_metrics'] = {}
    for alpha in alpha_values:
        cov, risk = compute_coverage_at_risk(
            pred_probs, y_test[:, :-1], selection_scores, alpha, loss_fn
        )
        results['id_metrics'][f'alpha_{alpha}'] = {
            'coverage': cov,
            'risk': risk
        }
    
    # Overall accuracy
    pred_labels = np.argmax(pred_probs, axis=1)
    true_labels = np.argmax(y_test[:, :-1], axis=1)
    results['overall_accuracy'] = np.mean(pred_labels == true_labels)
    
    # OOD metrics
    if x_ood is not None:
        ood_predictions, _ = model.predict(x_ood, batch_size=128)
        ood_selection_scores = ood_predictions[:, -1]
        
        ood_metrics = compute_ood_metrics(
            pred_probs, 
            ood_predictions[:, :-1],
            selection_scores,
            ood_selection_scores,
            selection_threshold=0.5
        )
        results['ood_metrics'] = ood_metrics
        results['ood_dataset'] = ood_name
    
    return results


def run_multiple_seeds(train_fn,
                      eval_fn,
                      seeds=[42, 43, 44, 45, 46],
                      **train_kwargs) -> Dict:
    """
    Run training and evaluation with multiple seeds for stability.
    
    Args:
        train_fn: function that trains and returns model
        eval_fn: function that evaluates model and returns metrics
        seeds: list of random seeds
        train_kwargs: kwargs for train_fn
    
    Returns:
        aggregated results with mean and std
    """
    all_results = []
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Running with seed {seed}")
        print(f"{'='*60}\n")
        
        # Set seed
        np.random.seed(seed)
        
        # Train
        model = train_fn(seed=seed, **train_kwargs)
        
        # Evaluate
        results = eval_fn(model)
        all_results.append(results)
    
    # Aggregate
    aggregated = aggregate_results(all_results)
    
    return aggregated


def aggregate_results(results_list: List[Dict]) -> Dict:
    """
    Aggregate results from multiple runs.
    
    Args:
        results_list: list of result dicts
    
    Returns:
        dict with mean and std for each metric
    """
    aggregated = {}
    
    # Get all metric paths
    def get_all_paths(d, prefix=''):
        paths = []
        for k, v in d.items():
            path = f"{prefix}/{k}" if prefix else k
            if isinstance(v, dict):
                paths.extend(get_all_paths(v, path))
            elif isinstance(v, (int, float)):
                paths.append(path)
        return paths
    
    all_paths = get_all_paths(results_list[0])
    
    # Aggregate each metric
    for path in all_paths:
        values = []
        for result in results_list:
            # Navigate to metric
            keys = path.split('/')
            val = result
            for k in keys:
                val = val[k]
            values.append(val)
        
        values = np.array(values)
        aggregated[path] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'values': values.tolist()
        }
    
    return aggregated


def save_results_json(results: Dict, filepath: str):
    """Save results to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        else:
            return obj
    
    results_converted = convert(results)
    
    with open(filepath, 'w') as f:
        json.dump(results_converted, f, indent=2)
    
    print(f"Saved results to {filepath}")


def generate_paper_figures(baseline_results,
                          crc_select_results,
                          output_dir='results/figures'):
    """
    Generate all figures for paper.
    
    Args:
        baseline_results: results from SelectiveNet baseline
        crc_select_results: results from CRC-Select
        output_dir: directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating paper figures in {output_dir}...")
    
    # Figure 1: RC curve
    rc_data = {
        'SelectiveNet': baseline_results.get('rc_curve', []),
        'CRC-Select': crc_select_results.get('rc_curve', [])
    }
    plot_rc_curve(rc_data, save_path=f"{output_dir}/rc_curve.png")
    
    # Figure 2: OOD comparison
    ood_data = {
        'SelectiveNet': baseline_results.get('ood_metrics', {}),
        'CRC-Select': crc_select_results.get('ood_metrics', {})
    }
    plot_ood_comparison(ood_data, save_path=f"{output_dir}/ood_comparison.png")
    
    # Table 1: Main results
    table_data = {
        'SelectiveNet': baseline_results.get('id_metrics', {}).get('alpha_0.05', {}),
        'CRC-Select': crc_select_results.get('id_metrics', {}).get('alpha_0.05', {})
    }
    create_results_table(table_data, save_path=f"{output_dir}/main_results.txt")
    
    print("Figures generated!")

