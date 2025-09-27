"""
Base model class for facial expression recognition
"""

import tensorflow as tf
from tensorflow.keras import Model, layers
from typing import Dict, List, Tuple, Optional
import abc


class BaseFacialExpressionModel(abc.ABC):
    """
    Abstract base class for facial expression recognition models
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 8, dropout_rate: float = 0.5):
        """
        Initialize the base model
        
        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of expression classes
            dropout_rate: Dropout rate for regularization
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.model = None
    
    @abc.abstractmethod
    def build_model(self) -> Model:
        """
        Build the model architecture
        
        Returns:
            Compiled Keras model
        """
        pass
    
    def get_model(self) -> Model:
        """
        Get the model instance
        
        Returns:
            Keras model
        """
        if self.model is None:
            self.model = self.build_model()
        return self.model
    
    def get_expression_head(self, features: tf.Tensor) -> tf.Tensor:
        """
        Get expression classification head
        
        Args:
            features: Feature tensor from backbone
            
        Returns:
            Expression predictions
        """
        x = layers.GlobalAveragePooling2D()(features)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        expression_output = layers.Dense(self.num_classes, activation='softmax', name='expression')(x)
        return expression_output
    
    def get_valence_head(self, features: tf.Tensor) -> tf.Tensor:
        """
        Get valence regression head
        
        Args:
            features: Feature tensor from backbone
            
        Returns:
            Valence predictions
        """
        x = layers.GlobalAveragePooling2D()(features)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        valence_output = layers.Dense(1, activation='tanh', name='valence')(x)
        return valence_output
    
    def get_arousal_head(self, features: tf.Tensor) -> tf.Tensor:
        """
        Get arousal regression head
        
        Args:
            features: Feature tensor from backbone
            
        Returns:
            Arousal predictions
        """
        x = layers.GlobalAveragePooling2D()(features)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        arousal_output = layers.Dense(1, activation='tanh', name='arousal')(x)
        return arousal_output
    
    def compile_model(self, learning_rate: float = 0.001) -> Model:
        """
        Compile the model with appropriate losses and metrics
        
        Args:
            learning_rate: Learning rate for optimizer
            
        Returns:
            Compiled model
        """
        model = self.get_model()
        
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
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )
        
        return model
    
    def get_model_summary(self) -> str:
        """
        Get model summary
        
        Returns:
            Model summary string
        """
        model = self.get_model()
        return model.summary()
    
    def save_model(self, filepath: str):
        """
        Save model to file
        
        Args:
            filepath: Path to save the model
        """
        model = self.get_model()
        model.save(filepath)
    
    def load_model(self, filepath: str):
        """
        Load model from file
        
        Args:
            filepath: Path to load the model from
        """
        self.model = tf.keras.models.load_model(filepath)



