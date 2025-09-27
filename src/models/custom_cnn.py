"""
Custom CNN architecture for facial expression recognition
"""

import tensorflow as tf
from tensorflow.keras import Model, layers
from .base_model import BaseFacialExpressionModel
from typing import Tuple


class CustomFacialExpressionModel(BaseFacialExpressionModel):
    """
    Custom CNN model for facial expression recognition with valence and arousal prediction
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 8, dropout_rate: float = 0.5):
        """
        Initialize custom CNN model
        
        Args:
            input_shape: Input image shape
            num_classes: Number of expression classes
            dropout_rate: Dropout rate
        """
        super().__init__(input_shape, num_classes, dropout_rate)
    
    def build_model(self) -> Model:
        """
        Build custom CNN model architecture
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = layers.Input(shape=self.input_shape, name='input')
        
        # Custom CNN backbone
        backbone = self._build_custom_backbone(inputs)
        
        # Multi-task heads
        expression_output = self.get_expression_head(backbone)
        valence_output = self.get_valence_head(backbone)
        arousal_output = self.get_arousal_head(backbone)
        
        # Create model
        model = Model(
            inputs=inputs,
            outputs=[expression_output, valence_output, arousal_output],
            name='Custom_CNN_FacialExpression'
        )
        
        return model
    
    def _build_custom_backbone(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Build custom CNN backbone with attention mechanisms
        
        Args:
            inputs: Input tensor
            
        Returns:
            Feature tensor
        """
        # Initial convolution
        x = layers.Conv2D(32, (7, 7), strides=(2, 2), padding='same', name='conv1')(inputs)
        x = layers.BatchNormalization(name='bn1')(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        
        # Block 1: 64 filters
        x = self._conv_block(x, 64, 3, stride=1, name='block1')
        x = self._conv_block(x, 64, 3, stride=1, name='block1_2')
        x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
        
        # Block 2: 128 filters with attention
        x = self._conv_block(x, 128, 3, stride=1, name='block2')
        x = self._conv_block(x, 128, 3, stride=1, name='block2_2')
        x = self._attention_block(x, 128, name='attention1')
        x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
        
        # Block 3: 256 filters with attention
        x = self._conv_block(x, 256, 3, stride=1, name='block3')
        x = self._conv_block(x, 256, 3, stride=1, name='block3_2')
        x = self._conv_block(x, 256, 3, stride=1, name='block3_3')
        x = self._attention_block(x, 256, name='attention2')
        x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
        
        # Block 4: 512 filters with attention
        x = self._conv_block(x, 512, 3, stride=1, name='block4')
        x = self._conv_block(x, 512, 3, stride=1, name='block4_2')
        x = self._conv_block(x, 512, 3, stride=1, name='block4_3')
        x = self._attention_block(x, 512, name='attention3')
        x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
        
        # Block 5: 1024 filters with attention
        x = self._conv_block(x, 1024, 3, stride=1, name='block5')
        x = self._conv_block(x, 1024, 3, stride=1, name='block5_2')
        x = self._attention_block(x, 1024, name='attention4')
        
        return x
    
    def _conv_block(self, x: tf.Tensor, filters: int, kernel_size: int, 
                   stride: int, name: str) -> tf.Tensor:
        """
        Build a convolutional block
        
        Args:
            x: Input tensor
            filters: Number of filters
            kernel_size: Kernel size
            stride: Stride
            name: Block name
            
        Returns:
            Output tensor
        """
        x = layers.Conv2D(filters, (kernel_size, kernel_size), strides=stride,
                         padding='same', name=f'{name}_conv')(x)
        x = layers.BatchNormalization(name=f'{name}_bn')(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(self.dropout_rate * 0.5)(x)
        
        return x
    
    def _attention_block(self, x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
        """
        Build an attention block
        
        Args:
            x: Input tensor
            filters: Number of filters
            name: Block name
            
        Returns:
            Output tensor with attention
        """
        # Channel attention
        channel_attention = layers.GlobalAveragePooling2D()(x)
        channel_attention = layers.Dense(filters // 4, activation='relu')(channel_attention)
        channel_attention = layers.Dense(filters, activation='sigmoid')(channel_attention)
        channel_attention = layers.Reshape((1, 1, filters))(channel_attention)
        
        # Spatial attention
        spatial_attention = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
        
        # Apply attention
        x = x * channel_attention * spatial_attention
        
        return x
    
    def build_lightweight_model(self) -> Model:
        """
        Build a lightweight version of the custom model
        
        Returns:
            Lightweight model
        """
        # Input layer
        inputs = layers.Input(shape=self.input_shape, name='input')
        
        # Lightweight backbone
        x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Multi-task heads
        expression_output = self.get_expression_head(x)
        valence_output = self.get_valence_head(x)
        arousal_output = self.get_arousal_head(x)
        
        # Create model
        model = Model(
            inputs=inputs,
            outputs=[expression_output, valence_output, arousal_output],
            name='Lightweight_Custom_CNN_FacialExpression'
        )
        
        return model
    
    def build_deep_model(self) -> Model:
        """
        Build a deeper version of the custom model
        
        Returns:
            Deep model
        """
        # Input layer
        inputs = layers.Input(shape=self.input_shape, name='input')
        
        # Deep backbone with residual connections
        x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        
        # Residual blocks
        x = self._residual_block(x, 64, 3, name='res1')
        x = self._residual_block(x, 64, 3, name='res2')
        
        x = self._residual_block(x, 128, 3, stride=2, name='res3')
        x = self._residual_block(x, 128, 3, name='res4')
        
        x = self._residual_block(x, 256, 3, stride=2, name='res5')
        x = self._residual_block(x, 256, 3, name='res6')
        
        x = self._residual_block(x, 512, 3, stride=2, name='res7')
        x = self._residual_block(x, 512, 3, name='res8')
        
        # Multi-task heads
        expression_output = self.get_expression_head(x)
        valence_output = self.get_valence_head(x)
        arousal_output = self.get_arousal_head(x)
        
        # Create model
        model = Model(
            inputs=inputs,
            outputs=[expression_output, valence_output, arousal_output],
            name='Deep_Custom_CNN_FacialExpression'
        )
        
        return model
    
    def _residual_block(self, x: tf.Tensor, filters: int, kernel_size: int, 
                       stride: int = 1, name: str = 'res') -> tf.Tensor:
        """
        Build a residual block
        
        Args:
            x: Input tensor
            filters: Number of filters
            kernel_size: Kernel size
            stride: Stride
            name: Block name
            
        Returns:
            Output tensor
        """
        shortcut = x
        
        # First convolution
        x = layers.Conv2D(filters, (kernel_size, kernel_size), strides=stride,
                         padding='same', name=f'{name}_conv1')(x)
        x = layers.BatchNormalization(name=f'{name}_bn1')(x)
        x = layers.Activation('relu')(x)
        
        # Second convolution
        x = layers.Conv2D(filters, (kernel_size, kernel_size), strides=1,
                         padding='same', name=f'{name}_conv2')(x)
        x = layers.BatchNormalization(name=f'{name}_bn2')(x)
        
        # Shortcut connection
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), strides=stride,
                                   padding='same', name=f'{name}_shortcut')(shortcut)
            shortcut = layers.BatchNormalization(name=f'{name}_shortcut_bn')(shortcut)
        
        # Add shortcut
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        
        return x



