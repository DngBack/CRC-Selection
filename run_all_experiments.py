"""
Run All Experiments - Full Comparison

This script runs:
1. SelectiveNet baseline
2. Post-hoc CRC
3. CRC-Select

With multiple seeds for stability analysis.
Then generates all paper figures and tables.
"""

import argparse
import numpy as np
import os
import json
from datetime import datetime

from run_baseline import run_baseline_experiment
from run_post_hoc_crc import run_post_hoc_crc
from run_crc_select import run_crc_select
from eval_utils import (
    plot_rc_curve,
    plot_ood_comparison,
    create_results_table,
    save_results_json,
    generate_paper_figures
)


def run_single_seed_all_methods(dataset='cifar_10',
                                exp_name='comparison',
                                alpha=0.05,
                                coverage=0.8,
                                ood_dataset='svhn',
                                seed=42,
                                run_baseline=True,
                                run_posthoc=True,
                                run_crcselect=True):
    """
    Run all three methods with a single seed.
    
    Returns:
        dict with results from all methods
    """
    print(f"\n{'='*80}")
    print(f"RUNNING ALL METHODS - SEED {seed}")
    print(f"{'='*80}\n")
    
    results = {}
    
    # Method 1: SelectiveNet Baseline
    if run_baseline:
        print(f"\n{'#'*80}")
        print(f"# METHOD 1: SelectiveNet Baseline (seed={seed})")
        print(f"{'#'*80}\n")
        
        baseline_results = run_baseline_experiment(
            dataset=dataset,
            model_name=f'{exp_name}_baseline_seed{seed}',
            alpha=0.5,  # SelectiveNet alpha (not CRC alpha)
            coverages=[coverage],  # Just one coverage for comparison
            ood_dataset=ood_dataset,
            seed=seed
        )
        results['selectivenet'] = baseline_results
    
    # Method 2: Post-hoc CRC
    if run_posthoc and run_baseline:
        print(f"\n{'#'*80}")
        print(f"# METHOD 2: Post-hoc CRC (seed={seed})")
        print(f"{'#'*80}\n")
        
        posthoc_results = run_post_hoc_crc(
            dataset=dataset,
            baseline_model=f'{exp_name}_baseline_seed{seed}',
            model_name=f'{exp_name}_posthoc_seed{seed}',
            coverage=coverage,
            alpha_values=[alpha],
            ood_dataset=ood_dataset,
            seed=seed
        )
        results['posthoc_crc'] = posthoc_results
    
    # Method 3: CRC-Select
    if run_crcselect:
        print(f"\n{'#'*80}")
        print(f"# METHOD 3: CRC-Select (seed={seed})")
        print(f"{'#'*80}\n")
        
        crcselect_results, _ = run_crc_select(
            dataset=dataset,
            model_name=f'{exp_name}_crcselect_seed{seed}',
            alpha=alpha,
            coverage_target=coverage,
            recalibrate_every=5,
            epochs=300,
            ood_dataset=ood_dataset,
            seed=seed
        )
        results['crc_select'] = crcselect_results
    
    return results


def aggregate_multiple_seeds(all_results):
    """
    Aggregate results from multiple seeds.
    
    Args:
        all_results: list of results dicts (one per seed)
    
    Returns:
        aggregated_results: dict with mean, std, etc.
    """
    aggregated = {}
    
    methods = ['selectivenet', 'posthoc_crc', 'crc_select']
    
    for method in methods:
        method_results = []
        for result in all_results:
            if method in result:
                method_results.append(result[method])
        
        if len(method_results) == 0:
            continue
        
        # Extract key metrics
        coverages = []
        risks = []
        accuracies = []
        dars = []
        
        for r in method_results:
            if method == 'selectivenet':
                # Get from first coverage result
                cov_key = list(r.keys())[0]
                if 'coverage_' in cov_key:
                    coverages.append(r[cov_key]['actual_coverage'])
                    risks.append(r[cov_key]['risk'])
                    accuracies.append(r[cov_key].get('accuracy', np.nan))
                if 'ood_metrics' in r:
                    dars.append(r['ood_metrics']['dar'])
            
            elif method == 'posthoc_crc':
                # Get from first alpha result
                alpha_key = list(r.keys())[0]
                if 'alpha_' in alpha_key:
                    coverages.append(r[alpha_key]['coverage'])
                    risks.append(r[alpha_key]['risk'])
                    accuracies.append(r[alpha_key]['accuracy'])
                if 'ood_metrics' in r:
                    dars.append(r['ood_metrics']['dar'])
            
            elif method == 'crc_select':
                coverages.append(r.get('test_coverage', np.nan))
                risks.append(r.get('test_risk', np.nan))
                accuracies.append(r.get('test_accuracy', np.nan))
                if 'ood_metrics' in r:
                    dars.append(r['ood_metrics']['dar'])
        
        # Aggregate
        aggregated[method] = {
            'coverage': {
                'mean': np.mean(coverages) if coverages else np.nan,
                'std': np.std(coverages) if coverages else np.nan,
                'values': coverages
            },
            'risk': {
                'mean': np.mean(risks) if risks else np.nan,
                'std': np.std(risks) if risks else np.nan,
                'values': risks
            },
            'accuracy': {
                'mean': np.mean(accuracies) if accuracies else np.nan,
                'std': np.std(accuracies) if accuracies else np.nan,
                'values': accuracies
            },
            'dar': {
                'mean': np.mean(dars) if dars else np.nan,
                'std': np.std(dars) if dars else np.nan,
                'values': dars
            },
            'n_seeds': len(coverages)
        }
    
    return aggregated


def generate_comparison_figures(aggregated_results, output_dir='results/figures'):
    """
    Generate all comparison figures for paper.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("Generating Comparison Figures")
    print(f"{'='*70}\n")
    
    # Figure 1: RC Curve Comparison
    print("Generating RC curve...")
    rc_data = {}
    for method, results in aggregated_results.items():
        cov_mean = results['coverage']['mean']
        risk_mean = results['risk']['mean']
        rc_data[method] = [(cov_mean, risk_mean)]
    
    plot_rc_curve(
        rc_data,
        save_path=f'{output_dir}/rc_curve_comparison.png',
        title='Risk-Coverage Comparison: All Methods'
    )
    
    # Figure 2: OOD Comparison
    print("Generating OOD comparison...")
    ood_data = {}
    for method, results in aggregated_results.items():
        ood_data[method] = {
            'dar': results['dar']['mean'],
            'id_coverage': results['coverage']['mean']
        }
    
    plot_ood_comparison(
        ood_data,
        save_path=f'{output_dir}/ood_comparison.png',
        title='OOD Dangerous Acceptance Rate Comparison'
    )
    
    # Table: Main Results
    print("Generating results table...")
    table_data = {}
    for method, results in aggregated_results.items():
        table_data[method] = {
            'Coverage': f"{results['coverage']['mean']:.3f} ± {results['coverage']['std']:.3f}",
            'Risk': f"{results['risk']['mean']:.4f} ± {results['risk']['std']:.4f}",
            'Accuracy': f"{results['accuracy']['mean']:.3f} ± {results['accuracy']['std']:.3f}",
            'DAR': f"{results['dar']['mean']:.3f} ± {results['dar']['std']:.3f}",
        }
    
    table_str = create_results_table(table_data, save_path=f'{output_dir}/main_results.txt')
    print("\nMain Results Table:")
    print(table_str)
    
    print(f"\nFigures saved to {output_dir}/")


def main(dataset='cifar_10',
         exp_name='comparison',
         alpha=0.05,
         coverage=0.8,
         ood_dataset='svhn',
         seeds=[42, 43, 44, 45, 46],
         run_baseline=True,
         run_posthoc=True,
         run_crcselect=True,
         skip_if_exists=True):
    """
    Run full experimental comparison.
    
    Args:
        dataset: dataset name
        exp_name: experiment name
        alpha: target risk level
        coverage: target coverage
        ood_dataset: OOD dataset
        seeds: list of random seeds
        run_baseline: whether to run baseline
        run_posthoc: whether to run post-hoc CRC
        run_crcselect: whether to run CRC-Select
        skip_if_exists: skip if results already exist
    """
    print(f"\n{'='*80}")
    print(f"FULL EXPERIMENTAL COMPARISON")
    print(f"{'='*80}")
    print(f"Dataset: {dataset}")
    print(f"Experiment: {exp_name}")
    print(f"Alpha: {alpha}")
    print(f"Coverage: {coverage}")
    print(f"OOD: {ood_dataset}")
    print(f"Seeds: {seeds}")
    print(f"Methods: ", end='')
    methods_to_run = []
    if run_baseline:
        methods_to_run.append('SelectiveNet')
    if run_posthoc:
        methods_to_run.append('Post-hoc CRC')
    if run_crcselect:
        methods_to_run.append('CRC-Select')
    print(', '.join(methods_to_run))
    print(f"{'='*80}\n")
    
    # Run all seeds
    all_results = []
    
    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'#'*80}")
        print(f"# SEED {seed_idx+1}/{len(seeds)}: {seed}")
        print(f"{'#'*80}\n")
        
        # Check if already done
        result_file = f'results/{exp_name}_seed{seed}_all.json'
        if skip_if_exists and os.path.exists(result_file):
            print(f"Results already exist for seed {seed}, loading...")
            with open(result_file, 'r') as f:
                results = json.load(f)
        else:
            # Run all methods
            results = run_single_seed_all_methods(
                dataset=dataset,
                exp_name=exp_name,
                alpha=alpha,
                coverage=coverage,
                ood_dataset=ood_dataset,
                seed=seed,
                run_baseline=run_baseline,
                run_posthoc=run_posthoc,
                run_crcselect=run_crcselect
            )
            
            # Save individual seed results
            save_results_json(results, result_file)
        
        all_results.append(results)
    
    # Aggregate results
    print(f"\n{'='*80}")
    print("AGGREGATING RESULTS ACROSS SEEDS")
    print(f"{'='*80}\n")
    
    aggregated = aggregate_multiple_seeds(all_results)
    
    # Save aggregated results
    aggregated_file = f'results/{exp_name}_aggregated.json'
    save_results_json(aggregated, aggregated_file)
    print(f"Aggregated results saved to {aggregated_file}")
    
    # Generate figures
    generate_comparison_figures(aggregated, output_dir=f'results/{exp_name}_figures')
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    for method, results in aggregated.items():
        print(f"{method.upper()}:")
        print(f"  Coverage: {results['coverage']['mean']:.2%} ± {results['coverage']['std']:.2%}")
        print(f"  Risk: {results['risk']['mean']:.4f} ± {results['risk']['std']:.4f}")
        print(f"  Accuracy: {results['accuracy']['mean']:.2%} ± {results['accuracy']['std']:.2%}")
        print(f"  DAR: {results['dar']['mean']:.2%} ± {results['dar']['std']:.2%}")
        print(f"  N seeds: {results['n_seeds']}")
        print()
    
    print(f"{'='*80}")
    print("EXPERIMENTAL COMPARISON COMPLETE!")
    print(f"{'='*80}\n")
    
    return aggregated


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run All Experiments')
    parser.add_argument('--dataset', type=str, default='cifar_10')
    parser.add_argument('--exp_name', type=str, default='comparison',
                       help='Experiment name for organizing results')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Target risk level')
    parser.add_argument('--coverage', type=float, default=0.8,
                       help='Target coverage')
    parser.add_argument('--ood', type=str, default='svhn')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                       help='List of seeds (e.g., --seeds 42 43 44)')
    parser.add_argument('--skip_baseline', action='store_true',
                       help='Skip baseline (if already run)')
    parser.add_argument('--skip_posthoc', action='store_true',
                       help='Skip post-hoc CRC (if already run)')
    parser.add_argument('--skip_crcselect', action='store_true',
                       help='Skip CRC-Select (if already run)')
    parser.add_argument('--no_skip_existing', action='store_true',
                       help='Rerun even if results exist')
    
    args = parser.parse_args()
    
    aggregated = main(
        dataset=args.dataset,
        exp_name=args.exp_name,
        alpha=args.alpha,
        coverage=args.coverage,
        ood_dataset=args.ood,
        seeds=args.seeds,
        run_baseline=not args.skip_baseline,
        run_posthoc=not args.skip_posthoc,
        run_crcselect=not args.skip_crcselect,
        skip_if_exists=not args.no_skip_existing
    )
    
    print("\n✓ All experiments completed!")
    print(f"  Check results/{args.exp_name}_figures/ for plots")
    print(f"  Check results/{args.exp_name}_aggregated.json for data")

