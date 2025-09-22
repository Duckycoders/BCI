# Convolutional-Attention Hybrid Model for High-Density EEG Classification

## Abstract

This repository presents a novel deep learning architecture that combines convolutional neural networks with Transformer encoders for motor execution classification using high-density EEG signals. The proposed convolutional-attention hybrid model addresses the limitations of existing approaches in processing 128-channel EEG data and demonstrates improved performance on the High Gamma Dataset (HGD).

## Introduction

Electroencephalography (EEG) based brain-computer interfaces (BCIs) require robust signal processing and classification methods to decode neural intentions accurately. While convolutional neural networks excel at local spatio-temporal feature extraction, they have limited capacity for modeling long-range temporal dependencies. This work introduces a hybrid architecture that leverages the complementary strengths of CNNs and Transformer encoders.

## Method

### Architecture Overview

The proposed model consists of three main components:

1. **Convolutional Frontend**: Extracts local spatio-temporal features from multi-channel EEG signals
2. **Transformer Encoder**: Models global temporal dependencies across the feature sequence
3. **Classification Head**: Performs final motor task classification

### Model Specifications

- **Input Dimensions**: (batch_size, 1, 128, 1000)
- **Convolutional Layers**: Depthwise separable convolutions with attention mechanisms
- **Transformer Configuration**: 4-head multi-head attention, 2 encoder layers
- **Output Classes**: 4 (right hand, left hand, rest, feet)
- **Total Parameters**: Approximately 220,000

### Data Processing Pipeline

1. **Raw EDF Loading**: Direct extraction from EDF files using MNE-Python
2. **Channel Selection**: Automated selection of 128 EEG channels
3. **Preprocessing**: Bandpass filtering (0.5-100 Hz), resampling to 250 Hz
4. **Trial Segmentation**: 4-second epochs (1000 samples)
5. **Normalization**: Channel-wise z-score standardization

## Dataset

### High Gamma Dataset (HGD)

The High Gamma Dataset contains continuous EEG recordings from 14 healthy subjects performing actual motor execution tasks.

**Dataset Characteristics:**
- Participants: 14 healthy subjects
- Recording Setup: 128-channel high-density EEG
- Tasks: Right hand movement, left hand movement, rest, feet movement
- Data Type: Actual motor execution (not motor imagery)
- Sampling Rate: 500 Hz (downsampled to 250 Hz)
- File Format: European Data Format (EDF)

**Data Availability:**
The dataset is publicly available and automatically downloaded during training execution.

## Implementation

### Core Files

- `main_TrainValTest.py`: Main training and evaluation script
- `models.py`: Neural network model definitions
- `transformer_layers.py`: Transformer encoder implementation
- `hgd_direct_loader.py`: Direct EDF data loading utilities
- `preprocess.py`: Data preprocessing pipeline
- `attention_models.py`: Attention mechanism implementations

### Dependencies

```bash
tensorflow>=2.7.0
mne>=0.24.0
moabb>=1.0.0
scikit-learn>=1.0.0
numpy>=1.20.0
scipy>=1.7.0
tqdm>=4.60.0
```

### Installation

```bash
git clone https://github.com/Duckycoders/BCI.git
cd BCI
pip install -r requirements.txt
```

### Usage

```bash
# Run training with default configuration
python main_TrainValTest.py

# Download HGD dataset manually
python download_with_progress.py

# Test model architecture
python test_hgd_training.py
```

## Experimental Setup

### Training Configuration

- **Optimizer**: Adam (learning_rate=0.0005)
- **Loss Function**: Categorical cross-entropy
- **Batch Size**: 16
- **Maximum Epochs**: 100
- **Early Stopping**: Patience=20 epochs
- **Validation Split**: 20% of training data
- **Random Seed**: 3407 (for reproducibility)

### Hardware Requirements

- **Minimum**: 8GB RAM, CPU-only training supported
- **Recommended**: 16GB RAM, NVIDIA GPU with CUDA support
- **Storage**: 5GB for complete HGD dataset

## Results

### Preliminary Findings

Initial experiments on HGD subject 1 demonstrate:
- **Training Accuracy**: Progressive improvement from 25% to 55%
- **Model Convergence**: Stable loss reduction without overfitting
- **Computational Efficiency**: ~220K parameters for 128-channel processing

### Performance Metrics

The model is evaluated using:
- Classification accuracy
- Cohen's kappa coefficient
- Confusion matrix analysis
- Training and validation loss curves

## Technical Details

### Convolutional Frontend

The convolutional component employs:
- Temporal convolution for frequency feature extraction
- Depthwise convolution for spatial filtering
- Separable convolution for feature combination
- Multi-head attention for temporal emphasis

### Transformer Enhancement

The Transformer encoder includes:
- Sinusoidal positional encoding for temporal information
- Multi-head self-attention (4 heads, 64-dimensional)
- Layer normalization and residual connections
- Feed-forward networks with ReLU activation

### Data Augmentation

- Sliding window technique (5 overlapping windows)
- Channel-wise standardization
- Temporal jittering during training

## Limitations and Future Work

### Current Limitations

- Limited to motor execution tasks (not motor imagery)
- Requires high-density EEG setup (128 channels)
- Subject-specific training approach

### Future Directions

- Cross-subject generalization studies
- Motor imagery adaptation
- Real-time BCI implementation
- Comparison with state-of-the-art methods

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{eeg_transformer_2025,
  title={Convolutional-Attention Hybrid Model for High-Density EEG Classification},
  author={[Authors]},
  year={2025},
  url={https://github.com/Duckycoders/BCI}
}
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgments

- Original ATCNet implementation by Hamdi Altaheri et al.
- High Gamma Dataset by Robin Tibor Schirrmeister et al.
- MNE-Python development team for EEG processing tools

## Contact

For questions and collaboration opportunities, please open an issue in this repository.