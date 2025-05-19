# GPT2-scratch

A simple implementation of GPT2 from scratch using PyTorch, designed for learning and educational purposes.

## Overview

This project implements a scaled-down version of GPT2 that trains on the TinyStories dataset. It includes key features like:

- Transformer-based architecture with self-attention
- Automatic Mixed Precision (AMP) training
- TensorBoard integration for monitoring training
- Checkpoint saving and loading
- Text generation capabilities

## Requirements

- Python 3.x
- PyTorch
- transformers
- datasets
- torch.utils.tensorboard
- tqdm

Install dependencies:

```bash
pip install torch transformers datasets tensorboard tqdm
```

## Usage

### Training

Run the training script:

```bash
python train.py
```

The script will:

1. Load and tokenize the TinyStories dataset (downloads automatically)
2. Initialize a GPT2 model with configurable parameters
3. Train using gradient accumulation and mixed precision
4. Save checkpoints during training
5. Log metrics to TensorBoard

### Configuration

Key hyperparameters in `train.py`:

- `batch_size`: 16 (effective batch size = batch_size * gradient_accumulation_steps)
- `block_size`: 256 (context window size)
- `learning_rate`: 3e-4
- `weight_decay`: 0.1
- Model architecture: 6 layers, 6 heads, 384 embedding dimensions

### Monitoring

Monitor training progress using TensorBoard:

```bash
tensorboard --logdir runs/gpt2_tinystories
```

### Text Generation

The model can generate text after training by loading the best checkpoint. Generation parameters like temperature and top-k sampling can be adjusted in the code.

## Model Architecture

The implementation includes:

- Token and positional embeddings
- Multi-head self-attention layers
- Layer normalization
- MLP blocks
- Causal attention mask for autoregressive generation

## License

MIT License
