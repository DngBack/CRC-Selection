"""
Run SelectiveNet Baseline Experiments

This script reproduces the original SelectiveNet results on:
- ID data (CIFAR-10)
- OOD evaluation (SVHN, CIFAR-100)
"""

import argparse
import numpy as np
import os

from models.cifar10_vgg_selectivenet import cifar10vgg
from models.svhn_vgg_selectivenet import SvhnVgg
from data_utils import load_ood_dataset
from eval_utils import (
    evaluate_model_full,
    save_results_json,
    plot_rc_curve,
    plot_ood_comparison
)
from crc_utils import compute_coverage_at_risk, compute_ood_metrics


def run_baseline_experiment(dataset='cifar_10',
                            model_name='baseline',
                            alpha=0.5,
                            coverages=[0.95, 0.9, 0.85, 0.8, 0.75, 0.7],
                            ood_dataset='svhn',
                            seed=42):
    """
    Run baseline SelectiveNet experiment.
    
    Args:
        dataset: 'cifar_10', 'svhn', 'catsdogs'
        model_name: name for saving
        alpha: alpha parameter for SelectiveNet (not CRC alpha!)
        coverages: target coverage rates to train
        ood_dataset: OOD dataset name
        seed: random seed
    """
    print(f"\n{'='*70}")
    print(f"Running SelectiveNet Baseline: {model_name}")
    print(f"Dataset: {dataset}, Alpha: {alpha}, OOD: {ood_dataset}")
    print(f"{'='*70}\n")
    
    np.random.seed(seed)
    
    # Select model class
    if dataset == 'cifar_10':
        model_cls = cifar10vgg
    elif dataset == 'svhn':
        model_cls = SvhnVgg
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Storage for results
    all_results = {}
    rc_points = []
    
    # Train for each coverage
    for coverage in coverages:
        print(f"\n{'='*50}")
        print(f"Training with target coverage = {coverage}")
        print(f"{'='*50}\n")
        
        filename = f"{model_name}_cov{coverage}.h5"
        
        # Check if already trained
        if os.path.exists(f"checkpoints/{filename}"):
            print(f"Loading existing model: {filename}")
            model = model_cls(train=False, filename=filename, coverage=coverage, alpha=alpha)
        else:
            print(f"Training new model: {filename}")
            model = model_cls(train=True, filename=filename, coverage=coverage, alpha=alpha)
        
        # Evaluate on ID test set
        print("\nEvaluating on ID test set...")
        predictions, _ = model.predict()
        pred_probs = predictions[:, :-1]
        pred_probs = pred_probs / (np.sum(pred_probs, axis=1, keepdims=True) + 1e-7)
        selection_scores = predictions[:, -1]
        
        # Compute risk and coverage
        from crc_utils import compute_risk_scores
        risk_scores = compute_risk_scores(pred_probs, model.y_test[:, :-1], 'cross_entropy')
        
        # At threshold 0.5
        accepted_mask = selection_scores >= 0.5
        actual_coverage = np.mean(accepted_mask)
        if np.sum(accepted_mask) > 0:
            actual_risk = np.mean(risk_scores[accepted_mask])
            accuracy = np.mean(
                np.argmax(pred_probs[accepted_mask], axis=1) == 
                np.argmax(model.y_test[accepted_mask, :-1], axis=1)
            )
        else:
            actual_risk = np.nan
            accuracy = np.nan
        
        print(f"Results: Coverage={actual_coverage:.2%}, Risk={actual_risk:.4f}, Accuracy={accuracy:.2%}")
        
        rc_points.append((actual_coverage, actual_risk))
        
        all_results[f'coverage_{coverage}'] = {
            'target_coverage': coverage,
            'actual_coverage': actual_coverage,
            'risk': actual_risk,
            'accuracy': accuracy
        }
    
    # Load OOD dataset
    print(f"\n{'='*50}")
    print(f"Loading OOD dataset: {ood_dataset}")
    print(f"{'='*50}\n")
    
    # Get normalization stats from model
    temp_model = model_cls(train=False, filename=f"{model_name}_cov{coverages[0]}.h5")
    mean = np.mean(temp_model.x_train, axis=(0, 1, 2, 3))
    std = np.std(temp_model.x_train, axis=(0, 1, 2, 3))
    
    x_ood = load_ood_dataset(ood_dataset, normalize_stats=(mean, std))
    
    if x_ood is not None:
        # Evaluate OOD on best model (middle coverage)
        best_cov_idx = len(coverages) // 2
        best_coverage = coverages[best_cov_idx]
        filename = f"{model_name}_cov{best_coverage}.h5"
        model = model_cls(train=False, filename=filename, coverage=best_coverage, alpha=alpha)
        
        print("\nEvaluating OOD metrics...")
        
        # ID predictions
        id_predictions, _ = model.predict()
        id_selection_scores = id_predictions[:, -1]
        
        # OOD predictions
        ood_predictions, _ = model.predict(x_ood, batch_size=128)
        ood_selection_scores = ood_predictions[:, -1]
        
        ood_metrics = compute_ood_metrics(
            id_predictions[:, :-1],
            ood_predictions[:, :-1],
            id_selection_scores,
            ood_selection_scores,
            selection_threshold=0.5
        )
        
        print(f"OOD Metrics:")
        print(f"  DAR (Dangerous Acceptance Rate): {ood_metrics['dar']:.2%}")
        print(f"  ID Coverage: {ood_metrics['id_coverage']:.2%}")
        print(f"  OOD Accepted: {ood_metrics['ood_accepted']}/{ood_metrics['ood_total']}")
        
        all_results['ood_metrics'] = ood_metrics
    
    # Save results
    all_results['rc_curve'] = rc_points
    all_results['config'] = {
        'dataset': dataset,
        'model_name': model_name,
        'alpha': alpha,
        'coverages': coverages,
        'ood_dataset': ood_dataset,
        'seed': seed
    }
    
    os.makedirs('results', exist_ok=True)
    save_results_json(all_results, f'results/{model_name}_results.json')
    
    # Plot RC curve
    plot_rc_curve(
        {'SelectiveNet': rc_points},
        save_path=f'results/{model_name}_rc_curve.png',
        title=f'SelectiveNet Risk-Coverage Curve ({dataset})'
    )
    
    print(f"\n{'='*70}")
    print(f"Baseline experiment completed! Results saved to results/{model_name}_results.json")
    print(f"{'='*70}\n")
    
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SelectiveNet Baseline')
    parser.add_argument('--dataset', type=str, default='cifar_10', 
                       help='Dataset: cifar_10, svhn, catsdogs')
    parser.add_argument('--model_name', type=str, default='baseline_selective',
                       help='Model name for saving')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Alpha for SelectiveNet training (0-1)')
    parser.add_argument('--ood', type=str, default='svhn',
                       help='OOD dataset: svhn, cifar100')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    results = run_baseline_experiment(
        dataset=args.dataset,
        model_name=args.model_name,
        alpha=args.alpha,
        ood_dataset=args.ood,
        seed=args.seed
    )
    
    print("\n✓ Baseline experiment finished!")
    print(f"  Check results/ folder for outputs")

