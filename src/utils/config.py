"""
Configuration management for facial expression recognition
"""

import os
from typing import Dict, Any
import json


class Config:
    """
    Configuration class for the facial expression recognition project
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._get_default_config()
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration
        
        Returns:
            Default configuration dictionary
        """
        return {
            # Dataset configuration
            'dataset': {
                'data_path': 'DL_Assignment1_Dataset/Dataset/Dataset',
                'image_size': (224, 224),
                'num_classes': 8,
                'test_size': 0.2,
                'val_size': 0.2,
                'random_state': 42
            },
            
            # Model configuration
            'model': {
                'input_shape': (224, 224, 3),
                'dropout_rate': 0.5,
                'use_pretrained': True,
                'freeze_backbone': True
            },
            
            # Training configuration
            'training': {
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
                'use_class_weights': True,
                'augmentation': True,
                'early_stopping_patience': 20,
                'lr_reduction_patience': 10
            },
            
            # Data augmentation
            'augmentation': {
                'horizontal_flip': True,
                'rotation_range': 10,
                'zoom_range': 0.1,
                'brightness_range': 0.2,
                'contrast_range': 0.1
            },
            
            # Paths
            'paths': {
                'results_dir': 'results',
                'models_dir': 'models',
                'logs_dir': 'logs',
                'plots_dir': 'plots'
            },
            
            # Evaluation
            'evaluation': {
                'metrics': ['accuracy', 'f1_score', 'cohen_kappa', 'auc', 'rmse', 'correlation', 'sagr', 'ccc'],
                'save_predictions': True,
                'save_plots': True
            }
        }
    
    def load_config(self, config_path: str):
        """
        Load configuration from file
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
        
        # Update configuration
        self._update_config(self.config, loaded_config)
    
    def save_config(self, config_path: str):
        """
        Save configuration to file
        
        Args:
            config_path: Path to save configuration
        """
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _update_config(self, base_config: Dict, update_config: Dict):
        """
        Recursively update configuration
        
        Args:
            base_config: Base configuration
            update_config: Update configuration
        """
        for key, value in update_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._update_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.get('paths.results_dir'),
            self.get('paths.models_dir'),
            self.get('paths.logs_dir'),
            self.get('paths.plots_dir')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get model-specific configuration
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model configuration
        """
        base_config = self.get('model', {})
        
        # Model-specific configurations
        model_configs = {
            'vgg16': {
                'vgg_type': 'vgg16',
                'pretrained_weights': 'imagenet'
            },
            'vgg19': {
                'vgg_type': 'vgg19',
                'pretrained_weights': 'imagenet'
            },
            'resnet50': {
                'resnet_type': 'resnet50',
                'pretrained_weights': 'imagenet'
            },
            'resnet101': {
                'resnet_type': 'resnet101',
                'pretrained_weights': 'imagenet'
            },
            'custom_cnn': {
                'use_attention': True,
                'use_residual': True
            }
        }
        
        model_config = model_configs.get(model_name, {})
        base_config.update(model_config)
        
        return base_config
    
    def get_training_config(self) -> Dict[str, Any]:
        """
        Get training configuration
        
        Returns:
            Training configuration
        """
        return self.get('training', {})
    
    def get_dataset_config(self) -> Dict[str, Any]:
        """
        Get dataset configuration
        
        Returns:
            Dataset configuration
        """
        return self.get('dataset', {})
    
    def get_augmentation_config(self) -> Dict[str, Any]:
        """
        Get augmentation configuration
        
        Returns:
            Augmentation configuration
        """
        return self.get('augmentation', {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """
        Get evaluation configuration
        
        Returns:
            Evaluation configuration
        """
        return self.get('evaluation', {})
    
    def print_config(self):
        """Print current configuration"""
        print("Current Configuration:")
        print(json.dumps(self.config, indent=2))



