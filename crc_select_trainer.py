"""
CRC-Select Training Module (Approach A: Alternating + Stop-Gradient)

This implements the core training loop for CRC-Select where:
1. Calibration step: compute CRC threshold q on calibration set (no gradient)
2. Training step: update model with q fixed, using risk penalty
3. Alternate between 1 and 2
"""

import numpy as np
import keras
from keras import backend as K
from keras.callbacks import Callback
import pickle
import os

from crc_utils import (
    compute_risk_scores,
    crc_calibrate,
    evaluate_crc
)


class CRCSelectCallback(Callback):
    """
    Custom callback for CRC-Select alternating optimization.
    
    This callback:
    - Runs CRC calibration every T epochs
    - Computes q on calibration set (stop-gradient)
    - Updates risk penalty weight (dual update)
    """
    
    def __init__(self, 
                 model_wrapper,
                 x_cal,
                 y_cal,
                 alpha=0.05,
                 recalibrate_every=5,
                 loss_fn='cross_entropy',
                 selection_threshold=0.5,
                 mu_init=1.0,
                 mu_lr=0.01,
                 verbose=True):
        """
        Args:
            model_wrapper: wrapper of the Keras model with helper methods
            x_cal: calibration images
            y_cal: calibration labels
            alpha: target risk level
            recalibrate_every: recalibrate q every T epochs
            loss_fn: loss function type
            selection_threshold: threshold for selection head
            mu_init: initial penalty weight
            mu_lr: learning rate for dual update of mu
            verbose: print logs
        """
        super().__init__()
        self.model_wrapper = model_wrapper
        self.x_cal = x_cal
        self.y_cal = y_cal
        self.alpha = alpha
        self.recalibrate_every = recalibrate_every
        self.loss_fn = loss_fn
        self.selection_threshold = selection_threshold
        self.mu = mu_init
        self.mu_lr = mu_lr
        self.verbose = verbose
        
        # History
        self.q_history = []
        self.mu_history = []
        self.cal_risk_history = []
        self.cal_coverage_history = []
        
        # Current q (will be updated)
        self.current_q = None
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        
        # Recalibrate every T epochs
        if epoch % self.recalibrate_every == 0:
            self._recalibrate()
        
        # Log
        if self.verbose and epoch % 5 == 0:
            print(f"\n[CRC-Select] Epoch {epoch}: q={self.current_q:.4f}, mu={self.mu:.4f}")
    
    def _recalibrate(self):
        """
        Calibration step: compute q on calibration set (stop-gradient).
        """
        # Predict on calibration set
        predictions, _ = self.model_wrapper.predict(self.x_cal, batch_size=128)
        
        # Extract predictions and selection scores
        # predictions shape: [N, num_classes + 1]
        # Last column is selection score
        pred_probs = predictions[:, :-1]  # [N, num_classes]
        selection_scores = predictions[:, -1]  # [N]
        
        # Normalize probabilities
        pred_probs = pred_probs / (np.sum(pred_probs, axis=1, keepdims=True) + 1e-7)
        
        # Compute risk scores
        risk_scores = compute_risk_scores(pred_probs, self.y_cal[:, :-1], self.loss_fn)
        
        # CRC calibration: find q
        q = crc_calibrate(
            risk_scores=risk_scores,
            selection_scores=selection_scores,
            alpha=self.alpha,
            selection_threshold=self.selection_threshold,
            lambda_param=0.01
        )
        
        # Update current q
        self.current_q = q
        self.q_history.append(q)
        
        # Compute metrics on calibration set
        accepted_mask = selection_scores >= self.selection_threshold
        cal_coverage = np.mean(accepted_mask)
        
        if np.sum(accepted_mask) > 0:
            cal_risk = np.mean(risk_scores[accepted_mask])
        else:
            cal_risk = np.nan
        
        self.cal_coverage_history.append(cal_coverage)
        self.cal_risk_history.append(cal_risk)
        
        # Dual update for mu (penalty weight)
        # If risk > alpha, increase mu; otherwise decrease
        if not np.isnan(cal_risk):
            violation = cal_risk - self.alpha
            self.mu = max(0.0, self.mu + self.mu_lr * violation)
        
        self.mu_history.append(self.mu)
        
        if self.verbose:
            print(f"\n[Calibration] q={q:.4f}, cal_risk={cal_risk:.4f}, "
                  f"cal_coverage={cal_coverage:.2%}, mu={self.mu:.4f}")
    
    def get_history(self):
        """Return calibration history."""
        return {
            'q': self.q_history,
            'mu': self.mu_history,
            'cal_risk': self.cal_risk_history,
            'cal_coverage': self.cal_coverage_history
        }


def create_crc_select_loss(num_classes, alpha, mu, lambda_coverage=32, loss_fn='cross_entropy'):
    """
    Create loss function for CRC-Select training.
    
    Loss = L_pred + lambda_cov * L_cov + mu * L_risk
    
    Args:
        num_classes: number of classes
        alpha: target risk level
        mu: penalty weight for risk constraint (will be updated by callback)
        lambda_coverage: weight for coverage regularizer
        loss_fn: loss function type
    
    Returns:
        loss function for selective head
    """
    
    # We'll use mu as a K.variable so it can be updated
    mu_var = K.variable(value=mu)
    
    def selective_loss(y_true, y_pred):
        """
        y_true: [batch, num_classes + 1] (last dim unused in this head)
        y_pred: [batch, num_classes + 1] (last dim is selection score g)
        """
        
        # Extract components
        y_true_class = y_true[:, :-1]  # [batch, num_classes]
        pred_class = y_pred[:, :-1]    # [batch, num_classes]
        selection_score = y_pred[:, -1:] # [batch, 1]
        
        # L_pred: selective cross-entropy
        # Weight each sample by its selection score (soft selection)
        weighted_labels = K.repeat_elements(selection_score, num_classes, axis=1) * y_true_class
        l_pred = K.categorical_crossentropy(weighted_labels, pred_class)
        l_pred = K.mean(l_pred)
        
        # L_cov: coverage regularizer
        # Penalize deviation from target coverage c
        # For now, we want to maximize coverage, so penalize low coverage
        mean_selection = K.mean(selection_score)
        target_coverage = 0.8  # Target 80% coverage
        l_cov = K.square(mean_selection - target_coverage)
        
        # L_risk: risk constraint penalty
        # This is a simplified version - in practice, we'd compute actual risk
        # Here we use a proxy: penalize high-risk predictions being accepted
        # High risk ~ low confidence
        max_prob = K.max(pred_class, axis=1, keepdims=True)
        confidence = max_prob
        risk_proxy = 1.0 - confidence  # Low confidence = high risk
        
        # Penalize accepting high-risk samples
        weighted_risk = selection_score * risk_proxy
        l_risk = K.mean(weighted_risk)
        
        # Total loss
        total_loss = l_pred + lambda_coverage * l_cov + mu_var * l_risk
        
        return total_loss
    
    return selective_loss, mu_var


def selective_acc(num_classes):
    """Selective accuracy metric."""
    def metric(y_true, y_pred):
        g = K.cast(K.greater(y_pred[:, -1], 0.5), K.floatx())
        temp1 = K.sum(
            (g) * K.cast(K.equal(K.argmax(y_true[:, :-1], axis=-1), 
                                  K.argmax(y_pred[:, :-1], axis=-1)), K.floatx()))
        temp1 = temp1 / (K.sum(g) + K.epsilon())
        return K.cast(temp1, K.floatx())
    return metric


def coverage_metric(y_true, y_pred):
    """Coverage metric."""
    g = K.cast(K.greater(y_pred[:, -1], 0.5), K.floatx())
    return K.mean(g)


class CRCSelectTrainer:
    """
    High-level trainer for CRC-Select.
    
    Usage:
        trainer = CRCSelectTrainer(model_cls, dataset='cifar_10')
        trainer.train(model_name='crc_select', alpha=0.05, epochs=300)
    """
    
    def __init__(self, model_cls, dataset='cifar_10', cal_ratio=0.2, seed=42):
        """
        Args:
            model_cls: model class (e.g., cifar10vgg)
            dataset: dataset name
            cal_ratio: calibration split ratio
            seed: random seed
        """
        self.model_cls = model_cls
        self.dataset = dataset
        self.cal_ratio = cal_ratio
        self.seed = seed
        
        # Will be set during training
        self.model = None
        self.x_train = None
        self.y_train = None
        self.x_cal = None
        self.y_cal = None
        self.x_test = None
        self.y_test = None
    
    def prepare_data(self):
        """Prepare train/cal/test splits."""
        from data_utils import split_train_calibration
        
        # Create a temporary model instance to load data
        temp_model = self.model_cls(train=False, filename='temp.h5')
        
        # Get full training data
        x_train_full = temp_model.x_train
        y_train_full = temp_model.y_train
        
        # Split into train and calibration
        self.x_train, self.y_train, self.x_cal, self.y_cal = split_train_calibration(
            x_train_full, y_train_full, self.cal_ratio, self.seed
        )
        
        # Test data
        self.x_test = temp_model.x_test
        self.y_test = temp_model.y_test
        
        print(f"Data prepared: train={len(self.x_train)}, cal={len(self.x_cal)}, test={len(self.x_test)}")
        
        del temp_model
    
    def train(self, model_name, alpha=0.05, epochs=300, recalibrate_every=5, mu_init=1.0):
        """
        Train CRC-Select model.
        
        Args:
            model_name: name for saving
            alpha: target risk level
            epochs: number of epochs
            recalibrate_every: recalibrate q every T epochs
            mu_init: initial penalty weight
        """
        print(f"\n{'='*60}")
        print(f"Training CRC-Select: {model_name}")
        print(f"Alpha={alpha}, Epochs={epochs}, Recalibrate every={recalibrate_every}")
        print(f"{'='*60}\n")
        
        # Prepare data if not done
        if self.x_train is None:
            self.prepare_data()
        
        # Create model instance
        # Note: we create the model without training (train=False)
        # then we'll train it manually with our custom loop
        self.model = self.model_cls(train=False, filename=f"{model_name}.h5")
        
        # We'll implement custom training in the next iteration
        # For now, placeholder
        print("[TODO] Custom CRC-Select training loop to be implemented")
        print("       Will use alternating optimization with CRC callback")
        
        return self.model


def save_crc_history(history, filename):
    """Save CRC calibration history."""
    os.makedirs('checkpoints', exist_ok=True)
    with open(f"checkpoints/{filename}_crc_history.pkl", 'wb') as f:
        pickle.dump(history, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved CRC history to checkpoints/{filename}_crc_history.pkl")

