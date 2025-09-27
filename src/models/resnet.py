"""
ResNet-based model for facial expression recognition
"""

import tensorflow as tf
from tensorflow.keras import Model, layers
from .base_model import BaseFacialExpressionModel
from typing import Tuple


class ResNetFacialExpressionModel(BaseFacialExpressionModel):
    """
    ResNet-based model for facial expression recognition with valence and arousal prediction
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 8, dropout_rate: float = 0.5,
                 resnet_type: str = 'resnet50'):
        """
        Initialize ResNet model
        
        Args:
            input_shape: Input image shape
            num_classes: Number of expression classes
            dropout_rate: Dropout rate
            resnet_type: Type of ResNet model ('resnet50', 'resnet101', 'resnet152')
        """
        super().__init__(input_shape, num_classes, dropout_rate)
        self.resnet_type = resnet_type
    
    def build_model(self) -> Model:
        """
        Build ResNet-based model architecture
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = layers.Input(shape=self.input_shape, name='input')
        
        # ResNet backbone
        if self.resnet_type == 'resnet50':
            backbone = self._build_resnet50_backbone(inputs)
        elif self.resnet_type == 'resnet101':
            backbone = self._build_resnet101_backbone(inputs)
        elif self.resnet_type == 'resnet152':
            backbone = self._build_resnet152_backbone(inputs)
        else:
            raise ValueError(f"Unsupported ResNet type: {self.resnet_type}")
        
        # Multi-task heads
        expression_output = self.get_expression_head(backbone)
        valence_output = self.get_valence_head(backbone)
        arousal_output = self.get_arousal_head(backbone)
        
        # Create model
        model = Model(
            inputs=inputs,
            outputs=[expression_output, valence_output, arousal_output],
            name=f'ResNet_{self.resnet_type}_FacialExpression'
        )
        
        return model
    
    def _build_resnet50_backbone(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Build ResNet50 backbone
        
        Args:
            inputs: Input tensor
            
        Returns:
            Feature tensor
        """
        # Initial convolution
        x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(inputs)
        x = layers.BatchNormalization(name='bn_conv1')(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        
        # ResNet blocks
        x = self._resnet_block(x, 64, 3, stride=1, stage=2, block='a')
        x = self._resnet_block(x, 64, 3, stride=1, stage=2, block='b')
        x = self._resnet_block(x, 64, 3, stride=1, stage=2, block='c')
        
        x = self._resnet_block(x, 128, 3, stride=2, stage=3, block='a')
        x = self._resnet_block(x, 128, 3, stride=1, stage=3, block='b')
        x = self._resnet_block(x, 128, 3, stride=1, stage=3, block='c')
        x = self._resnet_block(x, 128, 3, stride=1, stage=3, block='d')
        
        x = self._resnet_block(x, 256, 3, stride=2, stage=4, block='a')
        x = self._resnet_block(x, 256, 3, stride=1, stage=4, block='b')
        x = self._resnet_block(x, 256, 3, stride=1, stage=4, block='c')
        x = self._resnet_block(x, 256, 3, stride=1, stage=4, block='d')
        x = self._resnet_block(x, 256, 3, stride=1, stage=4, block='e')
        x = self._resnet_block(x, 256, 3, stride=1, stage=4, block='f')
        
        x = self._resnet_block(x, 512, 3, stride=2, stage=5, block='a')
        x = self._resnet_block(x, 512, 3, stride=1, stage=5, block='b')
        x = self._resnet_block(x, 512, 3, stride=1, stage=5, block='c')
        
        return x
    
    def _build_resnet101_backbone(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Build ResNet101 backbone
        
        Args:
            inputs: Input tensor
            
        Returns:
            Feature tensor
        """
        # Similar to ResNet50 but with more blocks in stage 3 and 4
        x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(inputs)
        x = layers.BatchNormalization(name='bn_conv1')(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        
        # ResNet blocks
        x = self._resnet_block(x, 64, 3, stride=1, stage=2, block='a')
        x = self._resnet_block(x, 64, 3, stride=1, stage=2, block='b')
        x = self._resnet_block(x, 64, 3, stride=1, stage=2, block='c')
        
        # Stage 3 with 22 blocks
        x = self._resnet_block(x, 128, 3, stride=2, stage=3, block='a')
        for i in range(21):
            x = self._resnet_block(x, 128, 3, stride=1, stage=3, block=chr(ord('b') + i))
        
        # Stage 4 with 35 blocks
        x = self._resnet_block(x, 256, 3, stride=2, stage=4, block='a')
        for i in range(34):
            x = self._resnet_block(x, 256, 3, stride=1, stage=4, block=chr(ord('b') + i))
        
        # Stage 5
        x = self._resnet_block(x, 512, 3, stride=2, stage=5, block='a')
        x = self._resnet_block(x, 512, 3, stride=1, stage=5, block='b')
        x = self._resnet_block(x, 512, 3, stride=1, stage=5, block='c')
        
        return x
    
    def _build_resnet152_backbone(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Build ResNet152 backbone
        
        Args:
            inputs: Input tensor
            
        Returns:
            Feature tensor
        """
        # Similar to ResNet101 but with even more blocks
        x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(inputs)
        x = layers.BatchNormalization(name='bn_conv1')(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        
        # ResNet blocks
        x = self._resnet_block(x, 64, 3, stride=1, stage=2, block='a')
        x = self._resnet_block(x, 64, 3, stride=1, stage=2, block='b')
        x = self._resnet_block(x, 64, 3, stride=1, stage=2, block='c')
        
        # Stage 3 with 35 blocks
        x = self._resnet_block(x, 128, 3, stride=2, stage=3, block='a')
        for i in range(34):
            x = self._resnet_block(x, 128, 3, stride=1, stage=3, block=chr(ord('b') + i))
        
        # Stage 4 with 35 blocks
        x = self._resnet_block(x, 256, 3, stride=2, stage=4, block='a')
        for i in range(34):
            x = self._resnet_block(x, 256, 3, stride=1, stage=4, block=chr(ord('b') + i))
        
        # Stage 5
        x = self._resnet_block(x, 512, 3, stride=2, stage=5, block='a')
        x = self._resnet_block(x, 512, 3, stride=1, stage=5, block='b')
        x = self._resnet_block(x, 512, 3, stride=1, stage=5, block='c')
        
        return x
    
    def _resnet_block(self, x: tf.Tensor, filters: int, kernel_size: int, 
                     stride: int, stage: int, block: str) -> tf.Tensor:
        """
        Build a ResNet block
        
        Args:
            x: Input tensor
            filters: Number of filters
            kernel_size: Kernel size
            stride: Stride
            stage: Stage number
            block: Block identifier
            
        Returns:
            Output tensor
        """
        conv_name_base = f'res{stage}{block}_branch'
        bn_name_base = f'bn{stage}{block}_branch'
        
        # Shortcut connection
        shortcut = x
        
        # First convolution
        x = layers.Conv2D(filters, (kernel_size, kernel_size), strides=stride,
                         padding='same', name=f'{conv_name_base}2a')(x)
        x = layers.BatchNormalization(name=f'{bn_name_base}2a')(x)
        x = layers.Activation('relu')(x)
        
        # Second convolution
        x = layers.Conv2D(filters, (kernel_size, kernel_size), strides=1,
                         padding='same', name=f'{conv_name_base}2b')(x)
        x = layers.BatchNormalization(name=f'{bn_name_base}2b')(x)
        
        # Shortcut connection
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), strides=stride,
                                   padding='same', name=f'{conv_name_base}1')(shortcut)
            shortcut = layers.BatchNormalization(name=f'{bn_name_base}1')(shortcut)
        
        # Add shortcut
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        
        return x
    
    def get_pretrained_resnet50(self) -> Model:
        """
        Get pre-trained ResNet50 model with custom heads
        
        Returns:
            Pre-trained ResNet50 model
        """
        # Load pre-trained ResNet50
        base_model = tf.keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze early layers
        for layer in base_model.layers[:-10]:
            layer.trainable = False
        
        # Add custom heads
        inputs = layers.Input(shape=self.input_shape, name='input')
        x = base_model(inputs, training=False)
        
        # Multi-task heads
        expression_output = self.get_expression_head(x)
        valence_output = self.get_valence_head(x)
        arousal_output = self.get_arousal_head(x)
        
        # Create model
        model = Model(
            inputs=inputs,
            outputs=[expression_output, valence_output, arousal_output],
            name='Pretrained_ResNet50_FacialExpression'
        )
        
        return model
    
    def get_pretrained_resnet101(self) -> Model:
        """
        Get pre-trained ResNet101 model with custom heads
        
        Returns:
            Pre-trained ResNet101 model
        """
        # Load pre-trained ResNet101
        base_model = tf.keras.applications.ResNet101(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze early layers
        for layer in base_model.layers[:-10]:
            layer.trainable = False
        
        # Add custom heads
        inputs = layers.Input(shape=self.input_shape, name='input')
        x = base_model(inputs, training=False)
        
        # Multi-task heads
        expression_output = self.get_expression_head(x)
        valence_output = self.get_valence_head(x)
        arousal_output = self.get_arousal_head(x)
        
        # Create model
        model = Model(
            inputs=inputs,
            outputs=[expression_output, valence_output, arousal_output],
            name='Pretrained_ResNet101_FacialExpression'
        )
        
        return model
