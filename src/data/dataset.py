"""
Dataset loading and preprocessing for facial expression recognition
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List, Optional
import glob


class FacialExpressionDataset:
    """
    Dataset class for facial expression recognition with valence and arousal prediction
    """
    
    def __init__(self, data_path: str, image_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the dataset
        
        Args:
            data_path: Path to the dataset directory
            image_size: Target image size (height, width)
        """
        self.data_path = data_path
        self.image_size = image_size
        self.images_path = os.path.join(data_path, "images")
        self.annotations_path = os.path.join(data_path, "annotations")
        
        # Expression labels mapping
        self.expression_labels = {
            0: "Neutral", 1: "Happy", 2: "Sad", 3: "Surprise",
            4: "Fear", 5: "Disgust", 6: "Anger", 7: "Contempt"
        }
        
        # Load all data
        self._load_data()
    
    def _load_data(self):
        """Load all images and annotations"""
        print("Loading dataset...")
        
        # Get all image files
        image_files = glob.glob(os.path.join(self.images_path, "*.jpg"))
        image_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
        
        self.image_paths = []
        self.expressions = []
        self.valences = []
        self.arousals = []
        self.landmarks = []
        
        for img_path in image_files:
            img_id = os.path.basename(img_path).split('.')[0]
            
            # Load annotations
            exp_path = os.path.join(self.annotations_path, f"{img_id}_exp.npy")
            val_path = os.path.join(self.annotations_path, f"{img_id}_val.npy")
            aro_path = os.path.join(self.annotations_path, f"{img_id}_aro.npy")
            lnd_path = os.path.join(self.annotations_path, f"{img_id}_lnd.npy")
            
            if all(os.path.exists(p) for p in [exp_path, val_path, aro_path, lnd_path]):
                self.image_paths.append(img_path)
                self.expressions.append(np.load(exp_path))
                self.valences.append(np.load(val_path))
                self.arousals.append(np.load(aro_path))
                self.landmarks.append(np.load(lnd_path))
        
        self.image_paths = np.array(self.image_paths)
        self.expressions = np.array(self.expressions, dtype=np.int32)
        self.valences = np.array(self.valences, dtype=np.float32)
        self.arousals = np.array(self.arousals, dtype=np.float32)
        self.landmarks = np.array(self.landmarks)
        
        print(f"Loaded {len(self.image_paths)} samples")
        print(f"Expression distribution: {np.bincount(self.expressions)}")
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess an image with improved preprocessing
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        image = Image.open(image_path).convert('RGB')
        image = image.resize(self.image_size)
        image = np.array(image, dtype=np.float32)
        
        # Improved preprocessing for better training
        # Normalize to [0, 1] first
        image = image / 255.0
        
        # Apply ImageNet normalization (better for pretrained models)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        return image
    
    def get_data_splits(self, test_size: float = 0.2, val_size: float = 0.2, 
                       random_state: int = 42) -> Dict[str, Dict]:
        """
        Split data into train, validation, and test sets
        
        Args:
            test_size: Proportion of data for test set
            val_size: Proportion of remaining data for validation set
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing train, val, test splits
        """
        # First split: train+val vs test
        X_temp, X_test, y_exp_temp, y_exp_test, y_val_temp, y_val_test, y_aro_temp, y_aro_test = train_test_split(
            self.image_paths, self.expressions, self.valences, self.arousals,
            test_size=test_size, random_state=random_state, stratify=self.expressions
        )
        
        # Second split: train vs val
        X_train, X_val, y_exp_train, y_exp_val, y_val_train, y_val_val, y_aro_train, y_aro_val = train_test_split(
            X_temp, y_exp_temp, y_val_temp, y_aro_temp,
            test_size=val_size/(1-test_size), random_state=random_state, stratify=y_exp_temp
        )
        
        return {
            'train': {
                'images': X_train,
                'expressions': y_exp_train,
                'valences': y_val_train,
                'arousals': y_aro_train
            },
            'val': {
                'images': X_val,
                'expressions': y_exp_val,
                'valences': y_val_val,
                'arousals': y_aro_val
            },
            'test': {
                'images': X_test,
                'expressions': y_exp_test,
                'valences': y_val_test,
                'arousals': y_aro_test
            }
        }
    
    def create_tf_dataset(self, data_split: Dict, batch_size: int = 32, 
                         shuffle: bool = True, augmentation: bool = False) -> tf.data.Dataset:
        """
        Create TensorFlow dataset
        
        Args:
            data_split: Data split dictionary
            batch_size: Batch size
            shuffle: Whether to shuffle data
            augmentation: Whether to apply data augmentation
            
        Returns:
            TensorFlow dataset
        """
        def load_and_preprocess(image_path, expression, valence, arousal):
            image = tf.py_function(
                lambda x: self.load_image(x.numpy().decode('utf-8')),
                [image_path], tf.float32
            )
            image.set_shape((*self.image_size, 3))
            
            if augmentation:
                image = self._apply_augmentation(image)
            
            return image, {
                'expression': expression,
                'valence': valence,
                'arousal': arousal
            }
        
        dataset = tf.data.Dataset.from_tensor_slices((
            data_split['images'],
            data_split['expressions'],
            data_split['valences'],
            data_split['arousals']
        ))
        
        dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _apply_augmentation(self, image: tf.Tensor) -> tf.Tensor:
        """
        Apply data augmentation to image
        
        Args:
            image: Input image tensor
            
        Returns:
            Augmented image tensor
        """
        # Random horizontal flip
        image = tf.image.random_flip_left_right(image)
        
        # Random rotation
        image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
        
        # Random brightness
        image = tf.image.random_brightness(image, max_delta=0.2)
        
        # Random contrast
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        
        # Random saturation
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        
        # Random zoom
        image = tf.image.random_crop(image, size=(*self.image_size, 3))
        
        return image
    
    def get_class_weights(self) -> np.ndarray:
        """
        Calculate class weights for imbalanced dataset
        
        Returns:
            Class weights array
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(self.expressions),
            y=self.expressions
        )
        
        return class_weights
    
    def get_dataset_stats(self) -> Dict:
        """
        Get dataset statistics
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'total_samples': len(self.image_paths),
            'image_size': self.image_size,
            'expression_distribution': dict(zip(
                [self.expression_labels[i] for i in range(8)],
                np.bincount(self.expressions)
            )),
            'valence_stats': {
                'mean': np.mean(self.valences),
                'std': np.std(self.valences),
                'min': np.min(self.valences),
                'max': np.max(self.valences)
            },
            'arousal_stats': {
                'mean': np.mean(self.arousals),
                'std': np.std(self.arousals),
                'min': np.min(self.arousals),
                'max': np.max(self.arousals)
            }
        }
        
        return stats
