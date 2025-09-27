# Facial Expression Recognition and Affect Analysis

This project implements CNN-based facial expression recognition and affect analysis using dimensional models of valence and arousal.

## Project Structure

```
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # Dataset loading and preprocessing
│   │   └── augmentation.py     # Data augmentation techniques
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py       # Base model class
│   │   ├── vgg.py             # VGG implementation
│   │   ├── resnet.py          # ResNet implementation
│   │   └── custom_cnn.py      # Custom CNN architecture
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py         # Training pipeline
│   │   └── metrics.py         # Evaluation metrics
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── visualization.py   # Plotting and visualization
│   │   └── config.py          # Configuration management
│   └── main.py               # Main execution script
├── notebooks/
│   └── facial_expression_analysis.ipynb  # Main Jupyter notebook
├── requirements.txt
└── README.md
```

## Dataset

The dataset contains:
- **Images**: 224x224 RGB face images
- **Annotations**: 
  - Expression labels (0-7): Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt
  - Valence values: [-1, +1] (negative to positive)
  - Arousal values: [-1, +1] (calm to excited)
  - 68 facial landmarks

## Features

- **Modular Architecture**: Clean, object-oriented design following best practices
- **Multiple CNN Baselines**: VGG, ResNet, and custom architectures
- **Comprehensive Evaluation**: Both categorical and continuous domain metrics
- **Data Augmentation**: On-the-fly and offline augmentation techniques
- **Transfer Learning**: Pre-trained model fine-tuning
- **Visualization**: Training curves and result analysis

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Jupyter Notebook
```bash
jupyter notebook notebooks/facial_expression_analysis.ipynb
```

### Command Line
```bash
python src/main.py --model vgg --epochs 50 --batch_size 32
```

## Evaluation Metrics

### Categorical Classification
- Accuracy, F1-Score, Cohen's Kappa, Krippendorff's Alpha
- AUC, AUC-PR

### Continuous Domain
- RMSE, Correlation, Sign Agreement Metric (SAGR)
- Concordance Correlation Coefficient (CCC)

## Results

Detailed results and analysis are provided in the Jupyter notebook and generated report.
