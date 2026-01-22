"""
Quick test script for debugging - runs with reduced epochs.
"""

import argparse
from run_baseline import run_baseline_experiment
from run_post_hoc_crc import run_post_hoc_crc
from run_crc_select import run_crc_select


def quick_test_baseline(dataset='cifar_10', epochs_override=10):
    """Quick test of baseline - just one coverage, few epochs."""
    print("\n" + "="*70)
    print("QUICK TEST: Baseline SelectiveNet")
    print("="*70 + "\n")
    
    # Temporarily modify epochs in model
    print(f"Running with {epochs_override} epochs (for quick test)")
    print("NOTE: This won't give meaningful results, just for debugging!\n")
    
    results = run_baseline_experiment(
        dataset=dataset,
        model_name='test_baseline',
        alpha=0.5,
        coverages=[0.8],  # Just one coverage
        ood_dataset='svhn',
        seed=42
    )
    
    print("\n✓ Quick baseline test completed!")
    return results


def quick_test_crc_select(dataset='cifar_10', epochs=10):
    """Quick test of CRC-Select."""
    print("\n" + "="*70)
    print("QUICK TEST: CRC-Select")
    print("="*70 + "\n")
    
    print(f"Running with {epochs} epochs (for quick test)")
    print("NOTE: This won't give meaningful results, just for debugging!\n")
    
    results, model = run_crc_select(
        dataset=dataset,
        model_name='test_crcselect',
        alpha=0.05,
        coverage_target=0.8,
        recalibrate_every=2,  # More frequent for short run
        epochs=epochs,
        seed=42
    )
    
    print("\n✓ Quick CRC-Select test completed!")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quick Test')
    parser.add_argument('--method', type=str, default='crcselect',
                       choices=['baseline', 'crcselect', 'all'],
                       help='Which method to test')
    parser.add_argument('--dataset', type=str, default='cifar_10')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs for quick test')
    
    args = parser.parse_args()
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                      QUICK TEST MODE                         ║
    ║                                                              ║
    ║  This runs with very few epochs for debugging purposes.     ║
    ║  Results will NOT be meaningful for research!               ║
    ║                                                              ║
    ║  For real experiments, use the full scripts:                ║
    ║    - run_baseline.py                                        ║
    ║    - run_crc_select.py                                      ║
    ║    - run_all_experiments.py                                 ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    if args.method == 'baseline' or args.method == 'all':
        quick_test_baseline(args.dataset, args.epochs)
    
    if args.method == 'crcselect' or args.method == 'all':
        quick_test_crc_select(args.dataset, args.epochs)
    
    print("\n" + "="*70)
    print("QUICK TEST COMPLETE")
    print("="*70)
    print("\nReminder: These are NOT real results!")
    print("Run full experiments for actual research results.")

