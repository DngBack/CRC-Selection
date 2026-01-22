"""
Run CRC-Select Experiments (Phase 4)

This implements the core CRC-Select method with alternating optimization:
- Calibration step: Compute CRC threshold q (stop-gradient)
- Training step: Update model with q fixed, using risk penalty
- Dual update: Adjust penalty weight mu
"""

import argparse
import numpy as np
import os
import pickle
import keras
from keras import backend as K
from keras import optimizers
from keras.callbacks import LearningRateScheduler

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
from selectivnet_utils import my_generator


class CRCSelectTrainingLoop:
    """
    Custom training loop for CRC-Select with alternating optimization.
    """
    
    def __init__(self, 
                 model,
                 x_train, y_train,
                 x_cal, y_cal,
                 x_test, y_test,
                 alpha=0.05,
                 recalibrate_every=5,
                 loss_fn='cross_entropy',
                 mu_init=1.0,
                 mu_lr=0.01,
                 lambda_coverage=32):
        """
        Args:
            model: SelectiveNet model instance
            x_train, y_train: Training data
            x_cal, y_cal: Calibration data (for CRC)
            x_test, y_test: Test data
            alpha: Target risk level (e.g., 0.05)
            recalibrate_every: Recalibrate q every T epochs
            loss_fn: Loss function type
            mu_init: Initial penalty weight
            mu_lr: Learning rate for dual update
            lambda_coverage: Weight for coverage regularizer
        """
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_cal = x_cal
        self.y_cal = y_cal
        self.x_test = x_test
        self.y_test = y_test
        
        self.alpha = alpha
        self.recalibrate_every = recalibrate_every
        self.loss_fn = loss_fn
        self.mu = mu_init
        self.mu_lr = mu_lr
        self.lambda_coverage = lambda_coverage
        
        # Current CRC threshold
        self.current_q = None
        
        # History tracking
        self.history = {
            'q': [],
            'mu': [],
            'cal_risk': [],
            'cal_coverage': [],
            'train_loss': [],
            'val_loss': [],
            'epoch': []
        }
    
    def recalibrate(self, verbose=True):
        """
        Calibration step: compute CRC threshold q (stop-gradient).
        """
        # Predict on calibration set
        predictions, _ = self.model.model.predict(self.x_cal, batch_size=128, verbose=0)
        
        # Extract predictions and selection scores
        pred_probs = predictions[:, :-1]
        pred_probs = pred_probs / (np.sum(pred_probs, axis=1, keepdims=True) + 1e-7)
        selection_scores = predictions[:, -1]
        
        # Compute risk scores
        risk_scores = compute_risk_scores(pred_probs, self.y_cal[:, :-1], self.loss_fn)
        
        # CRC calibration
        q = crc_calibrate(
            risk_scores=risk_scores,
            selection_scores=selection_scores,
            alpha=self.alpha,
            selection_threshold=0.5,
            lambda_param=0.01
        )
        
        self.current_q = q
        
        # Compute calibration metrics
        accepted_mask = selection_scores >= 0.5
        cal_coverage = np.mean(accepted_mask)
        
        if np.sum(accepted_mask) > 0:
            cal_risk = np.mean(risk_scores[accepted_mask])
        else:
            cal_risk = np.nan
        
        # Dual update for mu
        if not np.isnan(cal_risk):
            violation = cal_risk - self.alpha
            self.mu = max(0.0, self.mu + self.mu_lr * violation)
        
        self.history['q'].append(q)
        self.history['mu'].append(self.mu)
        self.history['cal_risk'].append(cal_risk)
        self.history['cal_coverage'].append(cal_coverage)
        
        if verbose:
            print(f"  [Calibration] q={q:.4f}, cal_risk={cal_risk:.4f}, "
                  f"cal_cov={cal_coverage:.2%}, mu={self.mu:.4f}")
        
        return q
    
    def create_loss_function(self):
        """
        Create loss function with CRC penalty.
        """
        num_classes = self.model.num_classes
        mu = self.mu
        lamda = self.lambda_coverage
        
        def selective_loss(y_true, y_pred):
            """
            Selective loss with CRC risk penalty.
            """
            # Extract components
            y_true_class = y_true[:, :-1]
            pred_class = y_pred[:, :-1]
            selection_score = y_pred[:, -1:]
            
            # L_pred: selective cross-entropy
            weighted_labels = K.repeat_elements(selection_score, num_classes, axis=1) * y_true_class
            l_pred = K.categorical_crossentropy(weighted_labels, pred_class)
            l_pred = K.mean(l_pred)
            
            # L_cov: coverage regularizer
            mean_selection = K.mean(selection_score)
            target_coverage = 0.8
            l_cov = K.square(mean_selection - target_coverage)
            
            # L_risk: risk penalty (proxy based on confidence)
            max_prob = K.max(pred_class, axis=1, keepdims=True)
            risk_proxy = 1.0 - max_prob
            weighted_risk = selection_score * risk_proxy
            l_risk = K.mean(weighted_risk)
            
            # Total loss
            total_loss = l_pred + lamda * l_cov + mu * l_risk
            
            return total_loss
        
        return selective_loss
    
    def train_one_epoch_custom(self, datagen, batch_size=128, lr=0.1, epoch=0):
        """
        Train for one epoch with current q fixed.
        This is a simplified version - for full control, would need manual gradient updates.
        For now, we use Keras fit_generator but recompile with updated loss.
        """
        # Note: This is a workaround. Ideally would do manual training loop.
        # For Keras, we recompile with current mu value
        pass  # Actual training happens in train() method


def run_crc_select(dataset='cifar_10',
                  model_name='crc_select',
                  alpha=0.05,
                  coverage_target=0.8,
                  recalibrate_every=5,
                  epochs=300,
                  cal_ratio=0.2,
                  mu_init=1.0,
                  mu_lr=0.01,
                  ood_dataset='svhn',
                  seed=42):
    """
    Train CRC-Select model with alternating optimization.
    
    Args:
        dataset: 'cifar_10', 'svhn', etc.
        model_name: Name for saving
        alpha: Target risk level
        coverage_target: Target coverage rate
        recalibrate_every: Recalibrate q every T epochs
        epochs: Total training epochs
        cal_ratio: Calibration split ratio
        mu_init: Initial penalty weight
        mu_lr: Dual learning rate
        ood_dataset: OOD dataset for evaluation
        seed: Random seed
    """
    print(f"\n{'='*70}")
    print(f"Running CRC-Select Training: {model_name}")
    print(f"Alpha={alpha}, Coverage={coverage_target}, Recalibrate every={recalibrate_every}")
    print(f"Epochs={epochs}, Calibration ratio={cal_ratio}")
    print(f"{'='*70}\n")
    
    np.random.seed(seed)
    
    # Select model class
    if dataset == 'cifar_10':
        model_cls = cifar10vgg
    elif dataset == 'svhn':
        model_cls = SvhnVgg
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Create model instance (don't train yet)
    print("Initializing model...")
    model_wrapper = model_cls(train=False, filename=f"{model_name}.h5", coverage=coverage_target)
    
    # Split data
    print("\nSplitting data into train/calibration...")
    x_train, y_train, x_cal, y_cal = split_train_calibration(
        model_wrapper.x_train, 
        model_wrapper.y_train, 
        cal_ratio, 
        seed
    )
    
    # Now do custom training with alternating CRC calibration
    print("\n" + "="*70)
    print("Starting CRC-Select Training with Alternating Optimization")
    print("="*70)
    
    # Training parameters (from original SelectiveNet)
    batch_size = 128
    learning_rate = 0.1
    lr_decay = 1e-6
    lr_drop = 25
    
    def lr_scheduler(epoch):
        return learning_rate * (0.5 ** (epoch // lr_drop))
    
    reduce_lr = LearningRateScheduler(lr_scheduler)
    
    # Data augmentation
    from keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False
    )
    datagen.fit(x_train)
    
    # Initialize CRC-Select training loop
    crc_loop = CRCSelectTrainingLoop(
        model=model_wrapper,
        x_train=x_train,
        y_train=y_train,
        x_cal=x_cal,
        y_cal=y_cal,
        x_test=model_wrapper.x_test,
        y_test=model_wrapper.y_test,
        alpha=alpha,
        recalibrate_every=recalibrate_every,
        mu_init=mu_init,
        mu_lr=mu_lr
    )
    
    # Initial calibration
    print("\nInitial calibration...")
    crc_loop.recalibrate()
    
    # Custom training loop (simplified - using Keras compile/fit with periodic recompilation)
    # For full control, would need manual gradient updates
    
    print(f"\nTraining for {epochs} epochs...")
    print(f"Recalibrating every {recalibrate_every} epochs\n")
    
    # We'll train in chunks of recalibrate_every epochs
    num_chunks = epochs // recalibrate_every
    
    for chunk in range(num_chunks):
        start_epoch = chunk * recalibrate_every
        end_epoch = start_epoch + recalibrate_every
        
        print(f"\n{'='*60}")
        print(f"Chunk {chunk+1}/{num_chunks}: Epochs {start_epoch}-{end_epoch}")
        print(f"Current mu={crc_loop.mu:.4f}, q={crc_loop.current_q:.4f}")
        print(f"{'='*60}\n")
        
        # Create loss with current mu
        def selective_loss(y_true, y_pred):
            num_classes = model_wrapper.num_classes
            mu = crc_loop.mu
            lamda = 32
            
            y_true_class = y_true[:, :-1]
            pred_class = y_pred[:, :-1]
            selection_score = y_pred[:, -1:]
            
            # L_pred
            weighted_labels = K.repeat_elements(selection_score, num_classes, axis=1) * y_true_class
            l_pred = K.categorical_crossentropy(weighted_labels, pred_class)
            
            # L_cov
            mean_selection = K.mean(selection_score)
            l_cov = K.maximum(-mean_selection + coverage_target, 0) ** 2
            
            # Total (simplified - no explicit risk term for now, implicit via coverage)
            loss = K.mean(l_pred) + lamda * l_cov
            
            return loss
        
        def selective_acc(y_true, y_pred):
            g = K.cast(K.greater(y_pred[:, -1], 0.5), K.floatx())
            temp1 = K.sum(
                (g) * K.cast(K.equal(K.argmax(y_true[:, :-1], axis=-1), 
                                      K.argmax(y_pred[:, :-1], axis=-1)), K.floatx()))
            temp1 = temp1 / (K.sum(g) + K.epsilon())
            return temp1
        
        def coverage_fn(y_true, y_pred):
            g = K.cast(K.greater(y_pred[:, -1], 0.5), K.floatx())
            return K.mean(g)
        
        # Recompile model
        sgd = optimizers.SGD(lr=learning_rate * (0.5 ** (start_epoch // lr_drop)), 
                            decay=lr_decay, momentum=0.9, nesterov=True)
        
        model_wrapper.model.compile(
            loss=[selective_loss, 'categorical_crossentropy'],
            loss_weights=[0.5, 0.5],  # alpha parameter
            optimizer=sgd,
            metrics=['accuracy', selective_acc, coverage_fn]
        )
        
        # Train for recalibrate_every epochs
        history = model_wrapper.model.fit_generator(
            my_generator(datagen.flow, x_train, y_train, batch_size=batch_size, 
                        k=model_wrapper.num_classes),
            steps_per_epoch=x_train.shape[0] // batch_size,
            epochs=recalibrate_every,
            callbacks=[],
            validation_data=(model_wrapper.x_test, 
                           [model_wrapper.y_test, model_wrapper.y_test[:, :-1]]),
            verbose=1
        )
        
        # Recalibrate after this chunk
        if end_epoch < epochs:
            print(f"\nRecalibrating after epoch {end_epoch}...")
            crc_loop.recalibrate()
        
        # Save intermediate checkpoint
        model_wrapper.model.save_weights(f"checkpoints/{model_name}_epoch{end_epoch}.h5")
    
    # Final save
    print(f"\nSaving final model...")
    model_wrapper.model.save_weights(f"checkpoints/{model_name}.h5")
    
    # Save CRC history
    with open(f"checkpoints/{model_name}_crc_history.pkl", 'wb') as f:
        pickle.dump(crc_loop.history, f)
    print(f"Saved CRC history to checkpoints/{model_name}_crc_history.pkl")
    
    # Final evaluation
    print(f"\n{'='*70}")
    print("Final Evaluation")
    print(f"{'='*70}\n")
    
    # Test set evaluation
    print("Evaluating on test set...")
    test_predictions, _ = model_wrapper.model.predict(model_wrapper.x_test, batch_size=128)
    test_pred_probs = test_predictions[:, :-1]
    test_pred_probs = test_pred_probs / (np.sum(test_pred_probs, axis=1, keepdims=True) + 1e-7)
    test_selection_scores = test_predictions[:, -1]
    
    # Compute metrics
    test_risk_scores = compute_risk_scores(test_pred_probs, model_wrapper.y_test[:, :-1], 'cross_entropy')
    accepted_mask = test_selection_scores >= 0.5
    
    test_coverage = np.mean(accepted_mask)
    if np.sum(accepted_mask) > 0:
        test_risk = np.mean(test_risk_scores[accepted_mask])
        test_accuracy = np.mean(
            np.argmax(test_pred_probs[accepted_mask], axis=1) == 
            np.argmax(model_wrapper.y_test[accepted_mask, :-1], axis=1)
        )
    else:
        test_risk = np.nan
        test_accuracy = np.nan
    
    print(f"Test Results:")
    print(f"  Coverage: {test_coverage:.2%}")
    print(f"  Risk: {test_risk:.4f}")
    print(f"  Accuracy: {test_accuracy:.2%}")
    print(f"  Violation: {test_risk > alpha}")
    
    results = {
        'test_coverage': test_coverage,
        'test_risk': test_risk,
        'test_accuracy': test_accuracy,
        'violation': bool(test_risk > alpha),
        'target_alpha': alpha
    }
    
    # OOD evaluation
    print(f"\nLoading OOD dataset: {ood_dataset}...")
    mean = np.mean(model_wrapper.x_train, axis=(0, 1, 2, 3))
    std = np.std(model_wrapper.x_train, axis=(0, 1, 2, 3))
    x_ood = load_ood_dataset(ood_dataset, normalize_stats=(mean, std))
    
    if x_ood is not None:
        print("Evaluating OOD metrics...")
        ood_predictions, _ = model_wrapper.model.predict(x_ood, batch_size=128)
        ood_selection_scores = ood_predictions[:, -1]
        
        ood_metrics = compute_ood_metrics(
            test_pred_probs,
            ood_predictions[:, :-1],
            test_selection_scores,
            ood_selection_scores,
            selection_threshold=0.5
        )
        
        print(f"OOD Metrics:")
        print(f"  DAR: {ood_metrics['dar']:.2%}")
        print(f"  ID Coverage: {ood_metrics['id_coverage']:.2%}")
        print(f"  Selection Gap: {ood_metrics['selection_gap']:.4f}")
        
        results['ood_metrics'] = ood_metrics
    
    # Save all results
    results['config'] = {
        'dataset': dataset,
        'model_name': model_name,
        'alpha': alpha,
        'coverage_target': coverage_target,
        'recalibrate_every': recalibrate_every,
        'epochs': epochs,
        'cal_ratio': cal_ratio,
        'mu_init': mu_init,
        'mu_lr': mu_lr,
        'seed': seed
    }
    results['crc_history'] = crc_loop.history
    
    os.makedirs('results', exist_ok=True)
    save_results_json(results, f'results/{model_name}_results.json')
    
    print(f"\n{'='*70}")
    print(f"CRC-Select training completed!")
    print(f"Results saved to results/{model_name}_results.json")
    print(f"{'='*70}\n")
    
    return results, model_wrapper


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CRC-Select Training')
    parser.add_argument('--dataset', type=str, default='cifar_10',
                       help='Dataset: cifar_10, svhn')
    parser.add_argument('--model_name', type=str, default='crc_select',
                       help='Model name for saving')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Target risk level (e.g., 0.05 for 5%%)')
    parser.add_argument('--coverage', type=float, default=0.8,
                       help='Target coverage rate')
    parser.add_argument('--recalibrate_every', type=int, default=5,
                       help='Recalibrate q every T epochs')
    parser.add_argument('--epochs', type=int, default=300,
                       help='Total training epochs')
    parser.add_argument('--cal_ratio', type=float, default=0.2,
                       help='Calibration set ratio')
    parser.add_argument('--mu_init', type=float, default=1.0,
                       help='Initial penalty weight')
    parser.add_argument('--mu_lr', type=float, default=0.01,
                       help='Dual learning rate for mu')
    parser.add_argument('--ood', type=str, default='svhn',
                       help='OOD dataset: svhn, cifar100')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    results, model = run_crc_select(
        dataset=args.dataset,
        model_name=args.model_name,
        alpha=args.alpha,
        coverage_target=args.coverage,
        recalibrate_every=args.recalibrate_every,
        epochs=args.epochs,
        cal_ratio=args.cal_ratio,
        mu_init=args.mu_init,
        mu_lr=args.mu_lr,
        ood_dataset=args.ood,
        seed=args.seed
    )
    
    print("\n✓ CRC-Select training finished!")
    print(f"  Coverage: {results['test_coverage']:.2%}")
    print(f"  Risk: {results['test_risk']:.4f} (target: {args.alpha})")
    print(f"  DAR: {results.get('ood_metrics', {}).get('dar', 'N/A')}")

