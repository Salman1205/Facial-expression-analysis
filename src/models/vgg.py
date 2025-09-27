"""
VGG-based model for facial expression recognition
"""

import tensorflow as tf
from tensorflow.keras import Model, layers
from .base_model import BaseFacialExpressionModel
from typing import Tuple


class VGGFacialExpressionModel(BaseFacialExpressionModel):
    """
    VGG-based model for facial expression recognition with valence and arousal prediction
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 8, dropout_rate: float = 0.5,
                 vgg_type: str = 'vgg16'):
        """
        Initialize VGG model
        
        Args:
            input_shape: Input image shape
            num_classes: Number of expression classes
            dropout_rate: Dropout rate
            vgg_type: Type of VGG model ('vgg16' or 'vgg19')
        """
        super().__init__(input_shape, num_classes, dropout_rate)
        self.vgg_type = vgg_type
    
    def build_model(self) -> Model:
        """
        Build VGG-based model architecture
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = layers.Input(shape=self.input_shape, name='input')
        
        # VGG backbone
        if self.vgg_type == 'vgg16':
            backbone = self._build_vgg16_backbone(inputs)
        elif self.vgg_type == 'vgg19':
            backbone = self._build_vgg19_backbone(inputs)
        else:
            raise ValueError(f"Unsupported VGG type: {self.vgg_type}")
        
        # Multi-task heads
        expression_output = self.get_expression_head(backbone)
        valence_output = self.get_valence_head(backbone)
        arousal_output = self.get_arousal_head(backbone)
        
        # Create model
        model = Model(
            inputs=inputs,
            outputs=[expression_output, valence_output, arousal_output],
            name=f'VGG_{self.vgg_type}_FacialExpression'
        )
        
        return model
    
    def _build_vgg16_backbone(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Build VGG16 backbone
        
        Args:
            inputs: Input tensor
            
        Returns:
            Feature tensor
        """
        # Block 1
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
        
        # Block 2
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
        
        # Block 3
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
        
        # Block 4
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
        
        # Block 5
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
        
        return x
    
    def _build_vgg19_backbone(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Build VGG19 backbone
        
        Args:
            inputs: Input tensor
            
        Returns:
            Feature tensor
        """
        # Block 1
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
        
        # Block 2
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
        
        # Block 3
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
        
        # Block 4
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
        
        # Block 5
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
        
        return x
    
    def get_pretrained_vgg16(self) -> Model:
        """
        Get pre-trained VGG16 model with custom heads
        
        Returns:
            Pre-trained VGG16 model
        """
        # Load pre-trained VGG16
        base_model = tf.keras.applications.VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze early layers
        for layer in base_model.layers[:-4]:
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
            name='Pretrained_VGG16_FacialExpression'
        )
        
        return model
    
    def get_pretrained_vgg19(self) -> Model:
        """
        Get pre-trained VGG19 model with custom heads
        
        Returns:
            Pre-trained VGG19 model
        """
        # Load pre-trained VGG19
        base_model = tf.keras.applications.VGG19(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze early layers
        for layer in base_model.layers[:-4]:
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
            name='Pretrained_VGG19_FacialExpression'
        )
        
        return model



