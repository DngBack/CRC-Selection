"""
Run Post-hoc CRC Baseline

This applies CRC calibration to an already-trained SelectiveNet model.
This is the "2-stage" baseline: train SelectiveNet, then apply CRC.
"""

import argparse
import numpy as np
import os

from models.cifar10_vgg_selectivenet import cifar10vgg
from models.svhn_vgg_selectivenet import SvhnVgg
from data_utils import split_train_calibration, load_ood_dataset
from crc_utils import (
    compute_risk_scores,
    crc_calibrate,
    evaluate_crc,
    compute_coverage_at_risk,
    compute_ood_metrics
)
from eval_utils import save_results_json, plot_rc_curve


def run_post_hoc_crc(dataset='cifar_10',
                     baseline_model='baseline_selective',
                     model_name='post_hoc_crc',
                     coverage=0.8,
                     alpha_values=[0.01, 0.05, 0.10],
                     cal_ratio=0.2,
                     ood_dataset='svhn',
                     seed=42):
    """
    Apply post-hoc CRC to trained SelectiveNet.
    
    Args:
        dataset: dataset name
        baseline_model: name of trained baseline model
        model_name: name for this experiment
        coverage: which coverage model to use
        alpha_values: list of CRC alpha (target risk) values to test
        cal_ratio: calibration split ratio
        ood_dataset: OOD dataset
        seed: random seed
    """
    print(f"\n{'='*70}")
    print(f"Running Post-hoc CRC: {model_name}")
    print(f"Using baseline: {baseline_model}, Coverage: {coverage}")
    print(f"CRC Alpha values: {alpha_values}")
    print(f"{'='*70}\n")
    
    np.random.seed(seed)
    
    # Select model class
    if dataset == 'cifar_10':
        model_cls = cifar10vgg
    elif dataset == 'svhn':
        model_cls = SvhnVgg
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Load trained baseline model
    filename = f"{baseline_model}_cov{coverage}.h5"
    if not os.path.exists(f"checkpoints/{filename}"):
        print(f"ERROR: Baseline model not found: checkpoints/{filename}")
        print(f"Please run run_baseline.py first!")
        return None
    
    print(f"Loading baseline model: {filename}")
    model = model_cls(train=False, filename=filename, coverage=coverage)
    
    # Split train into train + calibration
    print("\nSplitting data for calibration...")
    x_train, y_train, x_cal, y_cal = split_train_calibration(
        model.x_train, model.y_train, cal_ratio, seed
    )
    
    # Run CRC for each alpha
    all_results = {}
    rc_points = []
    
    for alpha in alpha_values:
        print(f"\n{'='*50}")
        print(f"CRC Calibration with alpha = {alpha}")
        print(f"{'='*50}\n")
        
        # Step 1: Calibrate on calibration set
        print("Step 1: Computing CRC threshold q on calibration set...")
        cal_predictions, _ = model.predict(x_cal, batch_size=128)
        cal_pred_probs = cal_predictions[:, :-1]
        cal_pred_probs = cal_pred_probs / (np.sum(cal_pred_probs, axis=1, keepdims=True) + 1e-7)
        cal_selection_scores = cal_predictions[:, -1]
        
        cal_risk_scores = compute_risk_scores(cal_pred_probs, y_cal[:, :-1], 'cross_entropy')
        
        q = crc_calibrate(
            risk_scores=cal_risk_scores,
            selection_scores=cal_selection_scores,
            alpha=alpha,
            selection_threshold=0.5,
            lambda_param=0.01
        )
        
        print(f"CRC threshold q = {q:.4f}")
        
        # Step 2: Evaluate on test set with this q
        print("Step 2: Evaluating on test set...")
        test_predictions, _ = model.predict(model.x_test, batch_size=128)
        test_pred_probs = test_predictions[:, :-1]
        test_pred_probs = test_pred_probs / (np.sum(test_pred_probs, axis=1, keepdims=True) + 1e-7)
        test_selection_scores = test_predictions[:, -1]
        
        metrics = evaluate_crc(
            predictions=test_pred_probs,
            labels=model.y_test[:, :-1],
            selection_scores=test_selection_scores,
            q=q,
            alpha=alpha,
            loss_fn='cross_entropy'
        )
        
        print(f"Test Metrics:")
        print(f"  Coverage: {metrics['coverage']:.2%}")
        print(f"  Risk: {metrics['risk']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.2%}")
        print(f"  Violation: {metrics['violation']}")
        
        rc_points.append((metrics['coverage'], metrics['risk']))
        
        all_results[f'alpha_{alpha}'] = metrics
    
    # OOD evaluation
    print(f"\n{'='*50}")
    print(f"OOD Evaluation on {ood_dataset}")
    print(f"{'='*50}\n")
    
    mean = np.mean(model.x_train, axis=(0, 1, 2, 3))
    std = np.std(model.x_train, axis=(0, 1, 2, 3))
    x_ood = load_ood_dataset(ood_dataset, normalize_stats=(mean, std))
    
    if x_ood is not None:
        # ID predictions
        id_predictions, _ = model.predict(model.x_test, batch_size=128)
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
        print(f"  DAR: {ood_metrics['dar']:.2%}")
        print(f"  ID Coverage: {ood_metrics['id_coverage']:.2%}")
        
        all_results['ood_metrics'] = ood_metrics
    
    # Save results
    all_results['rc_curve'] = rc_points
    all_results['config'] = {
        'dataset': dataset,
        'baseline_model': baseline_model,
        'model_name': model_name,
        'coverage': coverage,
        'alpha_values': alpha_values,
        'cal_ratio': cal_ratio,
        'ood_dataset': ood_dataset,
        'seed': seed
    }
    
    os.makedirs('results', exist_ok=True)
    save_results_json(all_results, f'results/{model_name}_results.json')
    
    # Plot RC curve
    plot_rc_curve(
        {'Post-hoc CRC': rc_points},
        save_path=f'results/{model_name}_rc_curve.png',
        title=f'Post-hoc CRC Risk-Coverage Curve ({dataset})'
    )
    
    print(f"\n{'='*70}")
    print(f"Post-hoc CRC completed! Results saved to results/{model_name}_results.json")
    print(f"{'='*70}\n")
    
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Post-hoc CRC')
    parser.add_argument('--dataset', type=str, default='cifar_10')
    parser.add_argument('--baseline', type=str, default='baseline_selective',
                       help='Name of trained baseline model')
    parser.add_argument('--model_name', type=str, default='post_hoc_crc')
    parser.add_argument('--coverage', type=float, default=0.8,
                       help='Which coverage model to use')
    parser.add_argument('--ood', type=str, default='svhn')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    results = run_post_hoc_crc(
        dataset=args.dataset,
        baseline_model=args.baseline,
        model_name=args.model_name,
        coverage=args.coverage,
        ood_dataset=args.ood,
        seed=args.seed
    )
    
    print("\n✓ Post-hoc CRC experiment finished!")

