"""
Main execution script for facial expression recognition
"""

import os
import sys
import argparse
from typing import Dict, Any

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset import FacialExpressionDataset
from models.vgg import VGGFacialExpressionModel
from models.resnet import ResNetFacialExpressionModel
from models.custom_cnn import CustomFacialExpressionModel
from training.trainer import FacialExpressionTrainer
from utils.config import Config


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Facial Expression Recognition Training')
    parser.add_argument('--model', type=str, default='vgg16', 
                       choices=['vgg16', 'vgg19', 'resnet50', 'resnet101', 'custom_cnn'],
                       help='Model architecture to train')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--data_path', type=str, 
                       default='DL_Assignment1_Dataset/Dataset/Dataset',
                       help='Path to dataset')
    parser.add_argument('--results_dir', type=str, default='results', 
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Facial Expression Recognition Training")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    
    # Load configuration
    config = Config()
    config.create_directories()
    
    # Load dataset
    print("\n📊 Loading dataset...")
    dataset = FacialExpressionDataset(args.data_path)
    data_splits = dataset.get_data_splits(test_size=0.2, val_size=0.2, random_state=42)
    
    # Create TensorFlow datasets
    train_dataset = dataset.create_tf_dataset(
        data_splits['train'], args.batch_size, shuffle=True, augmentation=True
    )
    val_dataset = dataset.create_tf_dataset(
        data_splits['val'], args.batch_size, shuffle=False, augmentation=False
    )
    test_dataset = dataset.create_tf_dataset(
        data_splits['test'], args.batch_size, shuffle=False, augmentation=False
    )
    
    # Initialize model
    print(f"\n🏗️ Building {args.model} model...")
    if args.model == 'vgg16':
        model = VGGFacialExpressionModel(vgg_type='vgg16')
        compiled_model = model.get_pretrained_vgg16()
    elif args.model == 'vgg19':
        model = VGGFacialExpressionModel(vgg_type='vgg19')
        compiled_model = model.get_pretrained_vgg19()
    elif args.model == 'resnet50':
        model = ResNetFacialExpressionModel(resnet_type='resnet50')
        compiled_model = model.get_pretrained_resnet50()
    elif args.model == 'resnet101':
        model = ResNetFacialExpressionModel(resnet_type='resnet101')
        compiled_model = model.get_pretrained_resnet101()
    elif args.model == 'custom_cnn':
        model = CustomFacialExpressionModel()
        compiled_model = model.build_model()
    
    # Compile model
    compiled_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss={
            'expression': 'sparse_categorical_crossentropy',
            'valence': 'mse',
            'arousal': 'mse'
        },
        loss_weights={'expression': 1.0, 'valence': 0.5, 'arousal': 0.5},
        metrics={
            'expression': ['accuracy'],
            'valence': ['mae'],
            'arousal': ['mae']
        }
    )
    
    # Initialize trainer
    trainer = FacialExpressionTrainer(compiled_model, args.model, args.results_dir)
    
    # Train model
    print(f"\n🚀 Training {args.model}...")
    history = trainer.train(
        train_dataset, val_dataset, 
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Evaluate model
    print(f"\n📊 Evaluating {args.model}...")
    metrics = trainer.evaluate(test_dataset)
    
    # Save model
    trainer.save_model()
    
    print(f"\n✅ Training completed successfully!")
    print(f"Results saved to: {args.results_dir}/{args.model}")


if __name__ == "__main__":
    main()



