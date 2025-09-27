"""
Data augmentation techniques for facial expression recognition
"""

import tensorflow as tf
from typing import Tuple, Dict, Any
import numpy as np


class DataAugmentation:
    """
    Data augmentation class for facial expression recognition
    """
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        """
        Initialize augmentation parameters
        
        Args:
            image_size: Target image size (height, width)
        """
        self.image_size = image_size
    
    def get_training_augmentation(self) -> tf.keras.Sequential:
        """
        Get training augmentation pipeline
        
        Returns:
            Keras Sequential model with augmentation layers
        """
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
            tf.keras.layers.RandomBrightness(0.1),
        ])
    
    def get_validation_augmentation(self) -> tf.keras.Sequential:
        """
        Get validation augmentation pipeline (minimal augmentation)
        
        Returns:
            Keras Sequential model with minimal augmentation
        """
        return tf.keras.Sequential([
            tf.keras.layers.Rescaling(1./255)
        ])
    
    def apply_offline_augmentation(self, images: np.ndarray, 
                                  labels: Dict[str, np.ndarray],
                                  augmentation_factor: int = 2) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Apply offline augmentation to increase dataset size
        
        Args:
            images: Input images array
            labels: Dictionary of labels
            augmentation_factor: Number of augmented versions per image
            
        Returns:
            Augmented images and labels
        """
        augmented_images = []
        augmented_labels = {key: [] for key in labels.keys()}
        
        for i in range(len(images)):
            # Original image
            augmented_images.append(images[i])
            for key in labels.keys():
                augmented_labels[key].append(labels[key][i])
            
            # Generate augmented versions
            for _ in range(augmentation_factor):
                # Apply random transformations
                img = tf.constant(images[i])
                img = tf.expand_dims(img, 0)
                
                # Random horizontal flip
                if tf.random.uniform([]) > 0.5:
                    img = tf.image.flip_left_right(img)
                
                # Random rotation
                angle = tf.random.uniform([], -0.1, 0.1)
                img = tf.image.rot90(img, k=tf.cast(angle * 4 / np.pi, tf.int32))
                
                # Random brightness
                img = tf.image.random_brightness(img, max_delta=0.2)
                
                # Random contrast
                img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
                
                # Random zoom
                img = tf.image.random_crop(img, size=(1, *self.image_size, 3))
                
                augmented_images.append(img.numpy()[0])
                for key in labels.keys():
                    augmented_labels[key].append(labels[key][i])
        
        return np.array(augmented_images), {key: np.array(augmented_labels[key]) for key in labels.keys()}
    
    def create_mixup_batch(self, images: tf.Tensor, labels: Dict[str, tf.Tensor], 
                          alpha: float = 0.2) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Apply Mixup augmentation to batch
        
        Args:
            images: Batch of images
            labels: Dictionary of labels
            alpha: Mixup parameter
            
        Returns:
            Mixed images and labels
        """
        batch_size = tf.shape(images)[0]
        lambda_param = tf.random.uniform([], 0, alpha)
        
        # Create random permutation
        indices = tf.random.shuffle(tf.range(batch_size))
        
        # Mix images
        mixed_images = lambda_param * images + (1 - lambda_param) * tf.gather(images, indices)
        
        # Mix labels
        mixed_labels = {}
        for key, label in labels.items():
            if key == 'expression':
                # For categorical labels, use soft labels
                mixed_labels[key] = lambda_param * label + (1 - lambda_param) * tf.gather(label, indices)
            else:
                # For continuous labels (valence, arousal), use linear interpolation
                mixed_labels[key] = lambda_param * label + (1 - lambda_param) * tf.gather(label, indices)
        
        return mixed_images, mixed_labels
    
    def create_cutmix_batch(self, images: tf.Tensor, labels: Dict[str, tf.Tensor],
                           alpha: float = 1.0) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Apply CutMix augmentation to batch
        
        Args:
            images: Batch of images
            labels: Dictionary of labels
            alpha: CutMix parameter
            
        Returns:
            CutMixed images and labels
        """
        batch_size = tf.shape(images)[0]
        lambda_param = tf.random.uniform([], 0, alpha)
        
        # Create random permutation
        indices = tf.random.shuffle(tf.range(batch_size))
        
        # Generate random bounding box
        h, w = tf.shape(images)[1], tf.shape(images)[2]
        cut_rat = tf.sqrt(1.0 - lambda_param)
        cut_h = tf.cast(h * cut_rat, tf.int32)
        cut_w = tf.cast(w * cut_rat, tf.int32)
        
        # Random center
        cx = tf.random.uniform([], 0, w, dtype=tf.int32)
        cy = tf.random.uniform([], 0, h, dtype=tf.int32)
        
        # Bounding box coordinates
        bbx1 = tf.clip_by_value(cx - cut_w // 2, 0, w)
        bby1 = tf.clip_by_value(cy - cut_h // 2, 0, h)
        bbx2 = tf.clip_by_value(cx + cut_w // 2, 0, w)
        bby2 = tf.clip_by_value(cy + cut_h // 2, 0, h)
        
        # Create mask
        mask = tf.zeros((h, w), dtype=tf.float32)
        mask = tf.tensor_scatter_nd_update(
            mask,
            tf.stack([tf.range(bby1, bby2)[:, None], tf.range(bbx1, bbx2)[None, :]], axis=2),
            tf.ones((bby2 - bby1, bbx2 - bbx1))
        )
        mask = tf.expand_dims(mask, axis=-1)
        mask = tf.expand_dims(mask, axis=0)
        
        # Apply CutMix
        mixed_images = images * (1 - mask) + tf.gather(images, indices) * mask
        
        # Adjust labels
        lambda_param = 1 - tf.reduce_sum(mask) / (h * w)
        mixed_labels = {}
        for key, label in labels.items():
            if key == 'expression':
                mixed_labels[key] = lambda_param * label + (1 - lambda_param) * tf.gather(label, indices)
            else:
                mixed_labels[key] = lambda_param * label + (1 - lambda_param) * tf.gather(label, indices)
        
        return mixed_images, mixed_labels



