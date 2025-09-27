"""
Visualization utilities for facial expression recognition
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class FacialExpressionVisualizer:
    """
    Visualization class for facial expression recognition results
    """
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        Initialize visualizer
        
        Args:
            style: Matplotlib style
        """
        plt.style.use(style)
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                      '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    def plot_dataset_distribution(self, dataset_stats: Dict, save_path: Optional[str] = None):
        """
        Plot dataset distribution
        
        Args:
            dataset_stats: Dataset statistics
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Dataset Distribution Analysis', fontsize=16)
        
        # Expression distribution
        expressions = list(dataset_stats['expression_distribution'].keys())
        counts = list(dataset_stats['expression_distribution'].values())
        
        axes[0, 0].bar(expressions, counts, color=self.colors[:len(expressions)])
        axes[0, 0].set_title('Expression Class Distribution')
        axes[0, 0].set_xlabel('Expression')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Valence distribution
        valence_mean = dataset_stats['valence_stats']['mean']
        valence_std = dataset_stats['valence_stats']['std']
        x = np.linspace(-1, 1, 100)
        y = np.exp(-0.5 * ((x - valence_mean) / valence_std) ** 2)
        axes[0, 1].plot(x, y, color='blue', linewidth=2)
        axes[0, 1].set_title('Valence Distribution')
        axes[0, 1].set_xlabel('Valence Value')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].grid(True)
        
        # Arousal distribution
        arousal_mean = dataset_stats['arousal_stats']['mean']
        arousal_std = dataset_stats['arousal_stats']['std']
        x = np.linspace(-1, 1, 100)
        y = np.exp(-0.5 * ((x - arousal_mean) / arousal_std) ** 2)
        axes[1, 0].plot(x, y, color='red', linewidth=2)
        axes[1, 0].set_title('Arousal Distribution')
        axes[1, 0].set_xlabel('Arousal Value')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].grid(True)
        
        # Valence vs Arousal scatter
        # Generate sample data for visualization
        n_samples = 1000
        valence_samples = np.random.normal(valence_mean, valence_std, n_samples)
        arousal_samples = np.random.normal(arousal_mean, arousal_std, n_samples)
        
        scatter = axes[1, 1].scatter(valence_samples, arousal_samples, 
                                    alpha=0.6, c=range(n_samples), cmap='viridis')
        axes[1, 1].set_title('Valence vs Arousal Scatter')
        axes[1, 1].set_xlabel('Valence')
        axes[1, 1].set_ylabel('Arousal')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_curves(self, history: Dict, model_name: str, save_path: Optional[str] = None):
        """
        Plot training curves
        
        Args:
            history: Training history
            model_name: Name of the model
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{model_name} Training Curves', fontsize=16)
        
        # Loss curves
        axes[0, 0].plot(history['loss'], label='Training Loss', color='blue')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Expression accuracy
        if 'expression_accuracy' in history:
            axes[0, 1].plot(history['expression_accuracy'], label='Training Accuracy', color='blue')
            axes[0, 1].plot(history['val_expression_accuracy'], label='Validation Accuracy', color='red')
            axes[0, 1].set_title('Expression Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Valence MAE
        if 'valence_mae' in history:
            axes[0, 2].plot(history['valence_mae'], label='Training MAE', color='blue')
            axes[0, 2].plot(history['val_valence_mae'], label='Validation MAE', color='red')
            axes[0, 2].set_title('Valence MAE')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('MAE')
            axes[0, 2].legend()
            axes[0, 2].grid(True)
        
        # Arousal MAE
        if 'arousal_mae' in history:
            axes[1, 0].plot(history['arousal_mae'], label='Training MAE', color='blue')
            axes[1, 0].plot(history['val_arousal_mae'], label='Validation MAE', color='red')
            axes[1, 0].set_title('Arousal MAE')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('MAE')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Learning rate
        if 'lr' in history:
            axes[1, 1].plot(history['lr'], color='green')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].grid(True)
        
        # Expression loss
        if 'expression_loss' in history:
            axes[1, 2].plot(history['expression_loss'], label='Training Loss', color='blue')
            axes[1, 2].plot(history['val_expression_loss'], label='Validation Loss', color='red')
            axes[1, 2].set_title('Expression Loss')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Loss')
            axes[1, 2].legend()
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             class_names: List[str], save_path: Optional[str] = None):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            save_path: Path to save plot
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - Expression Classification')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_valence_arousal_scatter(self, y_true_val: np.ndarray, y_pred_val: np.ndarray,
                                   y_true_aro: np.ndarray, y_pred_aro: np.ndarray,
                                   save_path: Optional[str] = None):
        """
        Plot valence and arousal scatter plots
        
        Args:
            y_true_val: True valence values
            y_pred_val: Predicted valence values
            y_true_aro: True arousal values
            y_pred_aro: Predicted arousal values
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Valence scatter
        axes[0].scatter(y_true_val, y_pred_val, alpha=0.6, color='blue')
        axes[0].plot([-1, 1], [-1, 1], 'r--', linewidth=2)
        axes[0].set_xlabel('True Valence')
        axes[0].set_ylabel('Predicted Valence')
        axes[0].set_title('Valence Prediction')
        axes[0].grid(True)
        
        # Calculate correlation
        val_corr = np.corrcoef(y_true_val, y_pred_val)[0, 1]
        axes[0].text(0.05, 0.95, f'Correlation: {val_corr:.3f}', 
                   transform=axes[0].transAxes, bbox=dict(boxstyle='round', facecolor='white'))
        
        # Arousal scatter
        axes[1].scatter(y_true_aro, y_pred_aro, alpha=0.6, color='red')
        axes[1].plot([-1, 1], [-1, 1], 'r--', linewidth=2)
        axes[1].set_xlabel('True Arousal')
        axes[1].set_ylabel('Predicted Arousal')
        axes[1].set_title('Arousal Prediction')
        axes[1].grid(True)
        
        # Calculate correlation
        aro_corr = np.corrcoef(y_true_aro, y_pred_aro)[0, 1]
        axes[1].text(0.05, 0.95, f'Correlation: {aro_corr:.3f}', 
                    transform=axes[1].transAxes, bbox=dict(boxstyle='round', facecolor='white'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, model_results: Dict[str, Dict], save_path: Optional[str] = None):
        """
        Plot model comparison
        
        Args:
            model_results: Dictionary of model results
            save_path: Path to save plot
        """
        models = list(model_results.keys())
        
        # Extract metrics
        expression_acc = [model_results[model]['expression']['accuracy'] for model in models]
        valence_rmse = [model_results[model]['valence']['rmse'] for model in models]
        arousal_rmse = [model_results[model]['arousal']['rmse'] for model in models]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Expression accuracy comparison
        axes[0].bar(models, expression_acc, color=self.colors[:len(models)])
        axes[0].set_title('Expression Accuracy Comparison')
        axes[0].set_ylabel('Accuracy')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Valence RMSE comparison
        axes[1].bar(models, valence_rmse, color=self.colors[:len(models)])
        axes[1].set_title('Valence RMSE Comparison')
        axes[1].set_ylabel('RMSE')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Arousal RMSE comparison
        axes[2].bar(models, arousal_rmse, color=self.colors[:len(models)])
        axes[2].set_title('Arousal RMSE Comparison')
        axes[2].set_ylabel('RMSE')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_sample_predictions(self, images: np.ndarray, y_true: Dict, y_pred: Dict,
                              class_names: List[str], num_samples: int = 8, 
                              save_path: Optional[str] = None):
        """
        Plot sample predictions
        
        Args:
            images: Sample images
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            num_samples: Number of samples to show
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Sample Predictions', fontsize=16)
        
        for i in range(min(num_samples, 8)):
            row = i // 4
            col = i % 4
            
            # Display image
            axes[row, col].imshow(images[i])
            axes[row, col].axis('off')
            
            # Get predictions
            true_exp = y_true['expression'][i]
            pred_exp = y_pred['expression'][i]
            true_val = y_true['valence'][i]
            pred_val = y_pred['valence'][i]
            true_aro = y_true['arousal'][i]
            pred_aro = y_pred['arousal'][i]
            
            # Create title
            title = f'True: {class_names[true_exp]}\nPred: {class_names[pred_exp]}\n'
            title += f'Val: {true_val:.2f}→{pred_val:.2f}\n'
            title += f'Aro: {true_aro:.2f}→{pred_aro:.2f}'
            
            # Color based on correctness
            color = 'green' if true_exp == pred_exp else 'red'
            axes[row, col].set_title(title, color=color, fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_plots(self, history: Dict, model_name: str):
        """
        Create interactive plots using Plotly
        
        Args:
            history: Training history
            model_name: Name of the model
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Loss', 'Expression Accuracy', 'Valence MAE',
                           'Arousal MAE', 'Learning Rate', 'Expression Loss'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Loss plot
        fig.add_trace(
            go.Scatter(y=history['loss'], name='Training Loss', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(y=history['val_loss'], name='Validation Loss', line=dict(color='red')),
            row=1, col=1
        )
        
        # Expression accuracy
        if 'expression_accuracy' in history:
            fig.add_trace(
                go.Scatter(y=history['expression_accuracy'], name='Training Accuracy', line=dict(color='blue')),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(y=history['val_expression_accuracy'], name='Validation Accuracy', line=dict(color='red')),
                row=1, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f'{model_name} Training Curves',
            showlegend=True,
            height=800,
            width=1200
        )
        
        fig.show()



