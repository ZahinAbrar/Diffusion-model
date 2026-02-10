# Simple Diffusion Model

A minimal implementation of a diffusion model for educational purposes, designed to run efficiently in Google Colab. This project demonstrates the core concepts of diffusion models using simple 2D geometric shapes.

## Overview

This implementation provides a clean, understandable introduction to diffusion models without the complexity of large-scale image generation. The model learns to generate simple geometric patterns (circles and squares) through a denoising diffusion process.

## Features

- **Simple U-Net Architecture**: Lightweight denoising network with skip connections
- **Cosine Noise Scheduling**: Stable noise schedule for better training dynamics  
- **Synthetic Dataset**: Procedurally generated circles and squares for fast training
- **Complete Pipeline**: Forward diffusion, training, and sampling implementation
- **Visualization Tools**: Built-in plotting for training loss and generated samples
- **Colab-Ready**: Optimized for quick experimentation in Google Colab

## Quick Start

### Google Colab (Recommended)
1. Open the notebook in Google Colab
2. Run all cells in sequence
3. The model will train for 20 epochs (~2-3 minutes on GPU)
4. Generated samples will be displayed automatically

### Local Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/simple-diffusion-model.git
cd simple-diffusion-model

# Install dependencies
pip install torch torchvision matplotlib numpy tqdm

# Run the model
python diffusion_model.py
```

## Model Architecture

### U-Net Denoising Network
- **Encoder**: Progressive downsampling with conv layers
- **Bottleneck**: Feature processing at lowest resolution
- **Decoder**: Upsampling with skip connections from encoder
- **Time Embedding**: Simple time conditioning for diffusion steps

### Diffusion Process
- **Forward Process**: Gradually adds Gaussian noise over T=300 timesteps
- **Reverse Process**: Learned denoising to generate samples from pure noise
- **Loss Function**: MSE between predicted and actual noise

## Training Details

- **Dataset**: 500 synthetic images (32×32 pixels)
- **Batch Size**: 16
- **Learning Rate**: 1e-3 (Adam optimizer)
- **Epochs**: 20 (adjustable)
- **Training Time**: ~2-3 minutes on GPU

## Results

The model successfully learns to generate recognizable geometric shapes after training. Generated samples show clear circles and squares with varying positions and sizes, demonstrating the model's ability to capture the training data distribution.

## Code Structure

```
diffusion_model.py
├── SimpleUNet class           # Neural network architecture
├── Noise scheduling          # Cosine beta schedule
├── Forward diffusion         # Noise addition process  
├── Training loop            # Model optimization
├── Sampling function        # Generation from noise
└── Visualization           # Results and comparisons
```

## Key Concepts Demonstrated

1. **Diffusion Process**: How noise is progressively added and removed
2. **Time Conditioning**: How the model learns time-dependent denoising
3. **U-Net Architecture**: Why skip connections matter for diffusion models
4. **Noise Scheduling**: Impact of different noise schedules on training stability

## Customization

### Modify Dataset
```python
# Change shape types or parameters
def create_simple_shapes(n_samples=1000, img_size=32):
    # Add triangles, different sizes, colors, etc.
```

### Adjust Model Parameters
```python
# Experiment with different architectures
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        # Modify layer sizes, add attention, etc.
```

### Training Configuration
```python
timesteps = 300      # Diffusion steps
epochs = 20          # Training epochs  
batch_size = 16      # Batch size
learning_rate = 1e-3 # Optimizer learning rate
```

## Educational Value

This implementation is ideal for:
- **Learning diffusion model fundamentals**
- **Understanding the training process**
- **Experimenting with hyperparameters**
- **Extending to more complex datasets**
- **Research prototyping**

## Requirements

- Python 3.7+
- PyTorch 1.9+
- torchvision
- matplotlib
- numpy
- tqdm

## Contributing

Contributions are welcome! Please feel free to submit pull requests for:
- Additional shape types or datasets
- Architecture improvements
- Visualization enhancements
- Documentation improvements

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This implementation is inspired by the foundational work on denoising diffusion probabilistic models:
- Ho et al. "Denoising Diffusion Probabilistic Models" (2020)
- Nichol & Dhariwal "Improved Denoising Diffusion Probabilistic Models" (2021)

## Citation

If you use this code in your research or educational materials, please cite:

```bibtex
@misc{simple-diffusion-model,
  title={Simple Diffusion Model: Educational Implementation},
  author={Abrar Zahin},
  year={2025},
  url={https://github.com/ZahinAbrar/diffusion-model}
}
```
