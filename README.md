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

## Dataset Setup

### Download Instructions

**⚠️ IMPORTANT:** The dataset is not included in this repository due to size constraints. You need to download it separately.

#### Option 1: Direct Download
1. Download the dataset from: [Dataset Download Link]
2. Extract the files to create the following structure:
```
DL_Assignment1_Dataset/
└── Dataset/
    └── Dataset/
        ├── images/
        │   ├── 0.jpg
        │   ├── 1.jpg
        │   └── ... (3,999 total images)
        └── annotations/
            ├── 0_exp.npy
            ├── 0_val.npy
            ├── 0_aro.npy
            ├── 0_lnd.npy
            └── ... (15,996 total annotation files)
```

#### Option 2: From Course Platform
- Download from your course assignment page
- Follow the same directory structure as shown above

### Dataset Information

The dataset contains:
- **Images**: 224x224 RGB face images (3,999 total)
- **Annotations**: 
  - Expression labels (0-7): Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt
  - Valence values: [-1, +1] (negative to positive)
  - Arousal values: [-1, +1] (calm to excited)
  - 68 facial landmarks per image

### Dataset Splits
- **Training**: 60% (2,399 samples)
- **Validation**: 20% (800 samples)  
- **Test**: 20% (800 samples)

## Features

- **Modular Architecture**: Clean, object-oriented design following best practices
- **Multiple CNN Baselines**: VGG, ResNet, and custom architectures
- **Comprehensive Evaluation**: Both categorical and continuous domain metrics
- **Data Augmentation**: On-the-fly and offline augmentation techniques
- **Transfer Learning**: Pre-trained model fine-tuning
- **Visualization**: Training curves and result analysis

## Installation

### Prerequisites
- Python 3.8+
- CUDA 12.3+ (for GPU support)


### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/Salman1205/Facial-expression-analysis.git
cd Facial-expression-analysis
```

2. **Create virtual environment:**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download and setup dataset:**
   - Follow the [Dataset Setup](#dataset-setup) instructions above
   - Place the dataset in the project root directory
   - Ensure the directory structure matches the expected format

5. **Verify GPU setup (optional):**
```bash
python -c "import tensorflow as tf; print(f'GPU available: {tf.config.list_physical_devices(\"GPU\")}')"
```

## Usage

### Quick Start (Recommended)
**Run the complete analysis using Jupyter Notebook:**
```bash
jupyter notebook notebooks/facial_expression_analysis.ipynb
```

The notebook includes:
- Dataset loading and visualization
- Model training (VGG16, ResNet50, Custom CNN)
- Performance evaluation and comparison
- Results visualization and analysis

### Command Line Interface
**Train individual models:**
```bash
# Train VGG16
python src/main.py --model vgg16 --epochs 50 --batch_size 32

# Train ResNet50
python src/main.py --model resnet50 --epochs 50 --batch_size 32

# Train Custom CNN
python src/main.py --model custom_cnn --epochs 50 --batch_size 32
```

**Available options:**
- `--model`: vgg16, resnet50, custom_cnn
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--data_path`: Path to dataset (default: DL_Assignment1_Dataset/Dataset/Dataset)

### Troubleshooting

**Common Issues:**
1. **"No GPU detected"**: Install CUDA 12.3+ and compatible TensorFlow
2. **"Dataset not found"**: Ensure dataset is in correct directory structure
3. **"Memory error"**: Reduce batch_size or use CPU-only version
4. **"Import errors"**: Check virtual environment is activated

**For GPU Issues:**
```bash
# Check CUDA installation
nvidia-smi

# Install GPU-enabled TensorFlow
pip install tensorflow[and-cuda]
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

## Report Template

A comprehensive report template is included for assignment submission:

- **File**: `report_template.md`
- **Format**: Markdown (convert to PDF for submission)
- **Sections**: Network details, training graphs, performance metrics, qualitative analysis
- **Screenshots**: Specific instructions for required figures and visualizations

**To convert to PDF:**
1. Open `report_template.md` in any markdown editor
2. Fill in your personal details and results
3. Add screenshots as specified in the template
4. Export to PDF with naming convention: `<FastID> <Name> A1-CS452.pdf`

## File Structure Notes

**Included in Repository:**
- Source code (`src/` directory)
- Jupyter notebook (`notebooks/`)
- Requirements and configuration files
- Report template (`report_template.md`)
- Results and plots (small files)

**NOT Included (Download Separately):**
- Dataset files (`DL_Assignment1_Dataset/`) - ~2-3 GB
- Trained model files (`*.h5`) - ~277 MB total
- Large output files

**Why Large Files Are Excluded:**
- GitHub has file size limits (100MB per file)
- Large files make repository slow to clone
- Dataset and models should be downloaded separately as needed

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is for educational purposes as part of CS452 Deep Learning course.

---

**Note**: This repository contains only the code and documentation. The dataset and trained models need to be downloaded separately due to size constraints.
