"""
CRC (Conformal Risk Control) Utilities for CRC-Select

This module implements the core CRC calibration logic used in CRC-Select.
Based on "Conformal Risk Control" (Angelopoulos et al., ICLR 2024)
"""

import numpy as np
from typing import Tuple, Optional, Callable


def compute_risk_scores(predictions: np.ndarray, 
                       labels: np.ndarray, 
                       loss_fn: str = 'cross_entropy') -> np.ndarray:
    """
    Compute per-sample risk scores.
    
    Args:
        predictions: [N, num_classes] prediction probabilities
        labels: [N, num_classes] one-hot labels
        loss_fn: type of loss ('cross_entropy', 'zero_one', 'margin')
    
    Returns:
        risk_scores: [N] per-sample risk scores
    """
    if loss_fn == 'cross_entropy':
        # Cross-entropy loss per sample
        # CE = -sum(y * log(p))
        epsilon = 1e-7
        pred_clipped = np.clip(predictions, epsilon, 1 - epsilon)
        risk_scores = -np.sum(labels * np.log(pred_clipped), axis=1)
        
    elif loss_fn == 'zero_one':
        # 0/1 loss: 1 if wrong, 0 if correct
        pred_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(labels, axis=1)
        risk_scores = (pred_labels != true_labels).astype(float)
        
    elif loss_fn == 'margin':
        # Margin-based surrogate loss
        # loss = max(0, margin - (p_true - p_max_other))
        true_labels = np.argmax(labels, axis=1)
        true_probs = predictions[np.arange(len(predictions)), true_labels]
        
        # Get max prob among other classes
        predictions_copy = predictions.copy()
        predictions_copy[np.arange(len(predictions)), true_labels] = -np.inf
        max_other_probs = np.max(predictions_copy, axis=1)
        
        margin = 0.1
        risk_scores = np.maximum(0, margin - (true_probs - max_other_probs))
        
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")
    
    return risk_scores


def crc_calibrate(risk_scores: np.ndarray,
                 selection_scores: np.ndarray,
                 alpha: float,
                 selection_threshold: float = 0.5,
                 lambda_param: float = 0.0) -> float:
    """
    CRC calibration: compute threshold q to control risk at level alpha.
    
    This implements the core CRC procedure:
    1. Filter to accepted samples (selection_score >= threshold)
    2. Compute empirical risk on accepted set
    3. Find q such that expected risk <= alpha
    
    Args:
        risk_scores: [N] per-sample risk scores on calibration set
        selection_scores: [N] selection scores g_phi(x) in [0,1]
        alpha: target risk level (e.g., 0.05 for 5% risk)
        selection_threshold: threshold for accepting (default 0.5)
        lambda_param: regularization for CRC (default 0.0)
    
    Returns:
        q: CRC threshold (higher q = more conservative)
    """
    # Get accepted indices
    accepted_mask = selection_scores >= selection_threshold
    
    if np.sum(accepted_mask) == 0:
        # No samples accepted - return very high threshold
        return np.inf
    
    # Risk scores on accepted set
    accepted_risks = risk_scores[accepted_mask]
    n_accepted = len(accepted_risks)
    
    # Sort risks
    sorted_risks = np.sort(accepted_risks)
    
    # CRC: find quantile q such that mean(risks <= q) controls risk
    # Simplified version: use quantile calibration
    # More sophisticated: use theorem from CRC paper
    
    # Method 1: Direct quantile (conservative)
    # We want: E[risk | accepted] <= alpha
    # Use (1-alpha)-quantile as threshold
    quantile_level = 1.0 - alpha
    q = np.quantile(sorted_risks, quantile_level)
    
    # Add small regularization to avoid edge cases
    q = q + lambda_param * np.std(accepted_risks)
    
    return q


def evaluate_crc(predictions: np.ndarray,
                labels: np.ndarray,
                selection_scores: np.ndarray,
                q: float,
                alpha: float,
                loss_fn: str = 'cross_entropy') -> dict:
    """
    Evaluate CRC metrics on test set.
    
    Args:
        predictions: [N, num_classes] predictions
        labels: [N, num_classes] one-hot labels
        selection_scores: [N] selection scores
        q: CRC threshold
        alpha: target risk level
        loss_fn: loss function type
    
    Returns:
        metrics: dict with coverage, risk, violation, etc.
    """
    # Compute risk scores
    risk_scores = compute_risk_scores(predictions, labels, loss_fn)
    
    # Accept based on selection score (could also incorporate q)
    selection_threshold = 0.5
    accepted_mask = selection_scores >= selection_threshold
    
    n_total = len(predictions)
    n_accepted = np.sum(accepted_mask)
    
    # Coverage
    coverage = n_accepted / n_total if n_total > 0 else 0.0
    
    # Risk on accepted set
    if n_accepted > 0:
        accepted_risks = risk_scores[accepted_mask]
        mean_risk = np.mean(accepted_risks)
        
        # Accuracy on accepted (for classification)
        pred_labels = np.argmax(predictions[accepted_mask], axis=1)
        true_labels = np.argmax(labels[accepted_mask], axis=1)
        accuracy = np.mean(pred_labels == true_labels)
    else:
        mean_risk = np.nan
        accuracy = np.nan
    
    # Violation: is risk > alpha?
    violation = mean_risk > alpha if not np.isnan(mean_risk) else False
    
    metrics = {
        'coverage': coverage,
        'risk': mean_risk,
        'accuracy': accuracy,
        'violation': violation,
        'violation_amount': max(0, mean_risk - alpha) if not np.isnan(mean_risk) else 0.0,
        'n_accepted': n_accepted,
        'n_total': n_total,
        'q_threshold': q
    }
    
    return metrics


def compute_coverage_at_risk(predictions: np.ndarray,
                            labels: np.ndarray,
                            selection_scores: np.ndarray,
                            target_risk: float,
                            loss_fn: str = 'cross_entropy') -> Tuple[float, float]:
    """
    Find maximum coverage that achieves target risk.
    
    This sweeps over selection thresholds to find the operating point.
    
    Args:
        predictions: [N, num_classes]
        labels: [N, num_classes]
        selection_scores: [N]
        target_risk: maximum acceptable risk
        loss_fn: loss function
    
    Returns:
        (coverage, actual_risk) at the operating point
    """
    risk_scores = compute_risk_scores(predictions, labels, loss_fn)
    
    # Try different thresholds
    thresholds = np.linspace(0, 1, 100)
    best_coverage = 0.0
    best_risk = np.inf
    
    for tau in thresholds:
        accepted_mask = selection_scores >= tau
        n_accepted = np.sum(accepted_mask)
        
        if n_accepted == 0:
            continue
        
        coverage = n_accepted / len(predictions)
        risk = np.mean(risk_scores[accepted_mask])
        
        # Check if risk constraint is satisfied
        if risk <= target_risk:
            if coverage > best_coverage:
                best_coverage = coverage
                best_risk = risk
    
    return best_coverage, best_risk


def compute_ood_metrics(id_predictions: np.ndarray,
                       ood_predictions: np.ndarray,
                       id_selection_scores: np.ndarray,
                       ood_selection_scores: np.ndarray,
                       selection_threshold: float = 0.5) -> dict:
    """
    Compute OOD-related metrics.
    
    Args:
        id_predictions: [N_id, num_classes] ID predictions
        ood_predictions: [N_ood, num_classes] OOD predictions
        id_selection_scores: [N_id] ID selection scores
        ood_selection_scores: [N_ood] OOD selection scores
        selection_threshold: threshold for acceptance
    
    Returns:
        metrics: dict with DAR, coverage, etc.
    """
    # Dangerous Acceptance Rate: fraction of OOD accepted
    ood_accepted = np.sum(ood_selection_scores >= selection_threshold)
    dar = ood_accepted / len(ood_selection_scores) if len(ood_selection_scores) > 0 else 0.0
    
    # ID coverage
    id_accepted = np.sum(id_selection_scores >= selection_threshold)
    id_coverage = id_accepted / len(id_selection_scores) if len(id_selection_scores) > 0 else 0.0
    
    # Average selection score
    id_avg_score = np.mean(id_selection_scores)
    ood_avg_score = np.mean(ood_selection_scores)
    
    metrics = {
        'dar': dar,  # Dangerous Acceptance Rate
        'id_coverage': id_coverage,
        'ood_accepted': ood_accepted,
        'ood_total': len(ood_selection_scores),
        'id_avg_selection': id_avg_score,
        'ood_avg_selection': ood_avg_score,
        'selection_gap': id_avg_score - ood_avg_score
    }
    
    return metrics


def compute_mixture_risk(id_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                        ood_data: Tuple[np.ndarray, np.ndarray],
                        ood_ratio: float,
                        selection_threshold: float = 0.5,
                        loss_fn: str = 'cross_entropy') -> dict:
    """
    Evaluate risk under ID + OOD mixture.
    
    Args:
        id_data: (predictions, labels, selection_scores) for ID
        ood_data: (predictions, selection_scores) for OOD (no labels)
        ood_ratio: fraction of OOD in mixture (0.0 to 1.0)
        selection_threshold: acceptance threshold
        loss_fn: loss function
    
    Returns:
        metrics: risk on accepted mixture
    """
    id_preds, id_labels, id_sel = id_data
    ood_preds, ood_sel = ood_data
    
    # Sample mixture
    n_id = len(id_preds)
    n_ood = len(ood_preds)
    n_ood_sample = int(ood_ratio * n_id / (1 - ood_ratio)) if ood_ratio < 1.0 else n_ood
    n_ood_sample = min(n_ood_sample, n_ood)
    
    # Random sample from OOD
    ood_indices = np.random.choice(n_ood, n_ood_sample, replace=False)
    
    # Accepted ID
    id_accepted_mask = id_sel >= selection_threshold
    id_accepted_count = np.sum(id_accepted_mask)
    
    # Accepted OOD (these are dangerous!)
    ood_accepted_mask = ood_sel[ood_indices] >= selection_threshold
    ood_accepted_count = np.sum(ood_accepted_mask)
    
    total_accepted = id_accepted_count + ood_accepted_count
    
    if total_accepted == 0:
        return {'mixture_risk': np.nan, 'coverage': 0.0, 'ood_contamination': 0.0}
    
    # Risk only defined on ID (we don't have OOD labels)
    # Assume OOD contributes maximum risk (e.g., always wrong)
    id_risk_scores = compute_risk_scores(
        id_preds[id_accepted_mask], 
        id_labels[id_accepted_mask], 
        loss_fn
    )
    
    # Pessimistic: assume OOD is always wrong (risk = 1.0 for 0/1 loss)
    # For cross-entropy, use high value
    ood_risk_value = 1.0 if loss_fn == 'zero_one' else 2.0
    
    total_risk = (np.sum(id_risk_scores) + ood_accepted_count * ood_risk_value) / total_accepted
    coverage = total_accepted / (n_id + n_ood_sample)
    ood_contamination = ood_accepted_count / total_accepted
    
    metrics = {
        'mixture_risk': total_risk,
        'coverage': coverage,
        'ood_contamination': ood_contamination,
        'id_accepted': id_accepted_count,
        'ood_accepted': ood_accepted_count
    }
    
    return metrics

