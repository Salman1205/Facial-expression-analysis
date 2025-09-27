"""
Training pipeline for facial expression recognition
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, 
    LearningRateScheduler, CSVLogger
)
from typing import Dict, List, Tuple, Optional, Callable
import matplotlib.pyplot as plt
import json
from datetime import datetime

from .metrics import FacialExpressionMetrics


class FacialExpressionTrainer:
    """
    Trainer class for facial expression recognition models
    """
    
    def __init__(self, model, model_name: str, results_dir: str = "results"):
        """
        Initialize trainer
        
        Args:
            model: Keras model to train
            model_name: Name of the model
            results_dir: Directory to save results
        """
        self.model = model
        self.model_name = model_name
        self.results_dir = results_dir
        self.metrics_calculator = FacialExpressionMetrics()
        
        # Force GPU usage
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        
        # Configure GPU memory growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ GPU configured for {model_name}")
            except RuntimeError as e:
                print(f"⚠️ GPU configuration warning: {e}")
        
        # Create results directory
        self.model_dir = os.path.join(results_dir, model_name)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Training history
        self.history = {}
        self.best_metrics = {}
    
    def train(self, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset,
              epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001,
              use_class_weights: bool = True, class_weights: Optional[Dict] = None,
              augmentation: bool = True) -> Dict:
        """
        Train the model
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            use_class_weights: Whether to use class weights
            class_weights: Class weights dictionary
            augmentation: Whether to use data augmentation
            
        Returns:
            Training history
        """
        print(f"Training {self.model_name}...")
        print(f"Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}")
        
        # Compile model
        self.model = self._compile_model(learning_rate, use_class_weights, class_weights)
        
        # Setup callbacks
        callbacks = self._setup_callbacks(epochs)
        
        # Train model with GPU
        start_time = datetime.now()
        with tf.device('/GPU:0'):
            history = self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        end_time = datetime.now()
        
        # Save training time
        training_time = (end_time - start_time).total_seconds()
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Store history
        self.history = history.history
        
        # Save training history
        self._save_training_history()
        
        # Plot training curves
        self._plot_training_curves()
        
        return self.history
    
    def evaluate(self, test_dataset: tf.data.Dataset) -> Dict[str, Dict[str, float]]:
        """
        Evaluate the model on test dataset
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Evaluation metrics
        """
        print(f"Evaluating {self.model_name}...")
        
        # Get predictions with GPU
        with tf.device('/GPU:0'):
            predictions = self.model.predict(test_dataset, verbose=1)
        
        # Extract true labels
        y_true = self._extract_labels_from_dataset(test_dataset)
        
        # Organize predictions
        y_pred = {
            'expression': np.argmax(predictions[0], axis=1),
            'valence': predictions[1].flatten(),
            'arousal': predictions[2].flatten()
        }
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_comprehensive_metrics(y_true, y_pred)
        
        # Save metrics
        self._save_metrics(metrics)
        
        # Print summary
        print(self.metrics_calculator.get_metric_summary(metrics))
        
        return metrics
    
    def _compile_model(self, learning_rate: float, use_class_weights: bool, 
                      class_weights: Optional[Dict]) -> tf.keras.Model:
        """
        Compile the model with appropriate settings
        
        Args:
            learning_rate: Learning rate
            use_class_weights: Whether to use class weights
            class_weights: Class weights dictionary
            
        Returns:
            Compiled model
        """
        # Define losses
        losses = {
            'expression': 'sparse_categorical_crossentropy',
            'valence': 'mse',
            'arousal': 'mse'
        }
        
        # Define loss weights
        loss_weights = {
            'expression': 1.0,
            'valence': 0.5,
            'arousal': 0.5
        }
        
        # Define metrics
        metrics = {
            'expression': ['accuracy'],
            'valence': ['mae'],
            'arousal': ['mae']
        }
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )
        
        return self.model
    
    def _setup_callbacks(self, epochs: int) -> List[tf.keras.callbacks.Callback]:
        """
        Setup training callbacks
        
        Args:
            epochs: Number of epochs
            
        Returns:
            List of callbacks
        """
        callbacks = []
        
        # Model checkpoint
        checkpoint_path = os.path.join(self.model_dir, 'best_model.h5')
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction
        lr_reduction = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(lr_reduction)
        
        # Learning rate scheduler
        lr_scheduler = LearningRateScheduler(
            self._lr_schedule,
            verbose=1
        )
        callbacks.append(lr_scheduler)
        
        # CSV logger
        csv_logger = CSVLogger(
            os.path.join(self.model_dir, 'training_log.csv'),
            append=False
        )
        callbacks.append(csv_logger)
        
        return callbacks
    
    def _lr_schedule(self, epoch: int, lr: float) -> float:
        """
        Learning rate schedule
        
        Args:
            epoch: Current epoch
            lr: Current learning rate
            
        Returns:
            New learning rate
        """
        if epoch < 10:
            return lr
        elif epoch < 30:
            return lr * 0.5
        elif epoch < 60:
            return lr * 0.1
        else:
            return lr * 0.01
    
    def _extract_labels_from_dataset(self, dataset: tf.data.Dataset) -> Dict[str, np.ndarray]:
        """
        Extract labels from dataset
        
        Args:
            dataset: TensorFlow dataset
            
        Returns:
            Dictionary of labels
        """
        expressions = []
        valences = []
        arousals = []
        
        for batch in dataset:
            expressions.extend(batch[1]['expression'].numpy())
            valences.extend(batch[1]['valence'].numpy())
            arousals.extend(batch[1]['arousal'].numpy())
        
        return {
            'expression': np.array(expressions),
            'valence': np.array(valences),
            'arousal': np.array(arousals)
        }
    
    def _save_training_history(self):
        """Save training history to file"""
        history_path = os.path.join(self.model_dir, 'training_history.json')
        
        # Convert numpy types to Python types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Convert history
        serializable_history = {}
        for key, values in self.history.items():
            serializable_history[key] = [convert_numpy(v) for v in values]
        
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
    
    def _plot_training_curves(self):
        """Plot and save training curves"""
        if not self.history:
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{self.model_name} Training Curves', fontsize=16)
        
        # Loss curves
        axes[0, 0].plot(self.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Expression accuracy
        if 'expression_accuracy' in self.history:
            axes[0, 1].plot(self.history['expression_accuracy'], label='Training Accuracy')
            axes[0, 1].plot(self.history['val_expression_accuracy'], label='Validation Accuracy')
            axes[0, 1].set_title('Expression Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Valence MAE
        if 'valence_mae' in self.history:
            axes[0, 2].plot(self.history['valence_mae'], label='Training MAE')
            axes[0, 2].plot(self.history['val_valence_mae'], label='Validation MAE')
            axes[0, 2].set_title('Valence MAE')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('MAE')
            axes[0, 2].legend()
            axes[0, 2].grid(True)
        
        # Arousal MAE
        if 'arousal_mae' in self.history:
            axes[1, 0].plot(self.history['arousal_mae'], label='Training MAE')
            axes[1, 0].plot(self.history['val_arousal_mae'], label='Validation MAE')
            axes[1, 0].set_title('Arousal MAE')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('MAE')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Learning rate
        if 'lr' in self.history:
            axes[1, 1].plot(self.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].grid(True)
        
        # Hide unused subplot
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.model_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_metrics(self, metrics: Dict[str, Dict[str, float]]):
        """Save evaluation metrics"""
        metrics_path = os.path.join(self.model_dir, 'evaluation_metrics.json')
        self.metrics_calculator.save_metrics(metrics, metrics_path)
    
    def save_model(self, filepath: Optional[str] = None):
        """
        Save the trained model
        
        Args:
            filepath: Path to save model (optional)
        """
        if filepath is None:
            filepath = os.path.join(self.model_dir, 'final_model.h5')
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model
        
        Args:
            filepath: Path to load model from
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> str:
        """Get model summary"""
        return self.model.summary()
    
    def compare_models(self, other_trainer: 'FacialExpressionTrainer') -> Dict:
        """
        Compare with another model
        
        Args:
            other_trainer: Another trainer instance
            
        Returns:
            Comparison results
        """
        comparison = {
            'model1': self.model_name,
            'model2': other_trainer.model_name,
            'model1_params': self.model.count_params(),
            'model2_params': other_trainer.model.count_params(),
            'model1_history': self.history,
            'model2_history': other_trainer.history
        }
        
        return comparison
