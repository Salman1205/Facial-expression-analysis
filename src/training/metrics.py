"""
Evaluation metrics for facial expression recognition
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, 
    roc_auc_score, average_precision_score,
    mean_squared_error, r2_score
)
from typing import Dict, List, Tuple, Optional
import warnings


class FacialExpressionMetrics:
    """
    Comprehensive metrics for facial expression recognition evaluation
    """
    
    def __init__(self):
        """Initialize metrics calculator"""
        self.metrics_history = {}
    
    def calculate_categorical_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate categorical classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # F1-Score (macro average)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='macro')
        
        # Cohen's Kappa
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # Krippendorff's Alpha (approximation)
        metrics['krippendorff_alpha'] = self._calculate_krippendorff_alpha(y_true, y_pred)
        
        # AUC metrics (if probabilities available)
        if y_pred_proba is not None:
            try:
                # Multi-class AUC
                metrics['auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
                
                # AUC-PR
                metrics['auc_pr'] = average_precision_score(y_true, y_pred_proba, average='macro')
            except:
                metrics['auc'] = 0.0
                metrics['auc_pr'] = 0.0
        
        return metrics
    
    def calculate_continuous_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate continuous domain metrics for valence and arousal
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # RMSE
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Correlation
        correlation = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
        metrics['correlation'] = correlation if not np.isnan(correlation) else 0.0
        
        # Sign Agreement Metric (SAGR)
        metrics['sagr'] = self._calculate_sagr(y_true, y_pred)
        
        # Concordance Correlation Coefficient (CCC)
        metrics['ccc'] = self._calculate_ccc(y_true, y_pred)
        
        # R² Score
        metrics['r2_score'] = r2_score(y_true, y_pred)
        
        # MAE
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        
        return metrics
    
    def _calculate_krippendorff_alpha(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Krippendorff's Alpha (simplified version)
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Krippendorff's Alpha value
        """
        try:
            # Convert to categorical
            unique_labels = np.unique(np.concatenate([y_true, y_pred]))
            n_categories = len(unique_labels)
            
            if n_categories < 2:
                return 0.0
            
            # Create confusion matrix
            confusion_matrix = np.zeros((n_categories, n_categories))
            for i, label in enumerate(unique_labels):
                true_mask = (y_true == label)
                for j, pred_label in enumerate(unique_labels):
                    pred_mask = (y_pred == pred_label)
                    confusion_matrix[i, j] = np.sum(true_mask & pred_mask)
            
            # Calculate agreement
            total = np.sum(confusion_matrix)
            if total == 0:
                return 0.0
            
            observed_agreement = np.trace(confusion_matrix) / total
            
            # Calculate expected agreement
            row_sums = np.sum(confusion_matrix, axis=1)
            col_sums = np.sum(confusion_matrix, axis=0)
            expected_agreement = np.sum(row_sums * col_sums) / (total ** 2)
            
            if expected_agreement >= 1.0:
                return 0.0
            
            alpha = (observed_agreement - expected_agreement) / (1 - expected_agreement)
            return max(0.0, alpha)
            
        except:
            return 0.0
    
    def _calculate_sagr(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Sign Agreement Metric (SAGR)
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            SAGR value
        """
        try:
            # Calculate sign agreement
            true_signs = np.sign(y_true)
            pred_signs = np.sign(y_pred)
            
            # Count correct signs
            correct_signs = np.sum(true_signs == pred_signs)
            total = len(y_true)
            
            return correct_signs / total if total > 0 else 0.0
            
        except:
            return 0.0
    
    def _calculate_ccc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Concordance Correlation Coefficient (CCC)
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            CCC value
        """
        try:
            y_true = y_true.flatten()
            y_pred = y_pred.flatten()
            
            # Calculate means
            mean_true = np.mean(y_true)
            mean_pred = np.mean(y_pred)
            
            # Calculate variances
            var_true = np.var(y_true)
            var_pred = np.var(y_pred)
            
            # Calculate covariance
            cov_true_pred = np.cov(y_true, y_pred)[0, 1]
            
            # Calculate CCC
            numerator = 2 * cov_true_pred
            denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
            
            if denominator == 0:
                return 0.0
            
            ccc = numerator / denominator
            return ccc
            
        except:
            return 0.0
    
    def calculate_comprehensive_metrics(self, y_true: Dict[str, np.ndarray], 
                                       y_pred: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Calculate comprehensive metrics for all tasks
        
        Args:
            y_true: Dictionary of true values for each task
            y_pred: Dictionary of predicted values for each task
            
        Returns:
            Dictionary of metrics for each task
        """
        results = {}
        
        # Expression classification metrics
        if 'expression' in y_true and 'expression' in y_pred:
            results['expression'] = self.calculate_categorical_metrics(
                y_true['expression'], y_pred['expression']
            )
        
        # Valence regression metrics
        if 'valence' in y_true and 'valence' in y_pred:
            results['valence'] = self.calculate_continuous_metrics(
                y_true['valence'], y_pred['valence']
            )
        
        # Arousal regression metrics
        if 'arousal' in y_true and 'arousal' in y_pred:
            results['arousal'] = self.calculate_continuous_metrics(
                y_true['arousal'], y_pred['arousal']
            )
        
        return results
    
    def get_metric_summary(self, metrics: Dict[str, Dict[str, float]]) -> str:
        """
        Get a formatted summary of metrics
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Formatted string summary
        """
        summary = "=== EVALUATION METRICS SUMMARY ===\n\n"
        
        for task, task_metrics in metrics.items():
            summary += f"--- {task.upper()} ---\n"
            for metric, value in task_metrics.items():
                summary += f"{metric}: {value:.4f}\n"
            summary += "\n"
        
        return summary
    
    def save_metrics(self, metrics: Dict[str, Dict[str, float]], filepath: str):
        """
        Save metrics to file
        
        Args:
            metrics: Dictionary of metrics
            filepath: Path to save metrics
        """
        import json
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Convert metrics
        serializable_metrics = {}
        for task, task_metrics in metrics.items():
            serializable_metrics[task] = {
                metric: convert_numpy(value) 
                for metric, value in task_metrics.items()
            }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
    
    def load_metrics(self, filepath: str) -> Dict[str, Dict[str, float]]:
        """
        Load metrics from file
        
        Args:
            filepath: Path to load metrics from
            
        Returns:
            Dictionary of metrics
        """
        import json
        
        with open(filepath, 'r') as f:
            metrics = json.load(f)
        
        return metrics



