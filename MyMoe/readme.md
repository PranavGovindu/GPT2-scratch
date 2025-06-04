# MyMoE:Mini Deepseek Mixture of Experts Implementation



## Architecture Overview

This implementation features a transformer-based architecture with MoE layers, incorporating several modern techniques:

- **Mixture of Experts Layer**: Implements a sparse MoE layer with configurable number of experts and routing mechanisms
- **SwiGLU Activation**: Uses SwiGLU activation in the feed-forward networks for improved performance
- **RoPE Positional Encoding**: Implements Rotary Positional Embeddings (RoPE) for better sequence modeling
- **RMSNorm**: Uses Root Mean Square Layer Normalization for improved training stability
- **Gradient Checkpointing**: Implements memory-efficient training through gradient checkpointing

### Key Components

1. **MoE Layer (`MoE.py`)**
   - Configurable number of experts
   - Top-k routing mechanism
   - Capacity factor for load balancing
   - Router noise for improved training
   - Auxiliary loss for expert utilization

2. **Transformer Block (`Transformer_Block.py`)**
   - Multi-head attention mechanism
   - MoE integration
   - Residual connections
   - Layer normalization

3. **Attention Mechanism (`Attention.py`)**
   - Multi-head attention implementation
   - RoPE integration
   - KV-head optimization

## Implementation Details

### Model Configuration (`config.py`)

Key configuration parameters:
- Vocabulary size: 50304
- Embedding dimension: 768
- Number of layers: 12
- Number of attention heads: 4
- Number of KV heads: 2
- Number of experts: 2
- Top-k routing: 1
- FFN dimension multiplier: 2.0
- Maximum sequence length: 128

### Training Configuration (`training_config.py`)

Training-specific parameters and optimization settings.

## Usage

### Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Training

To train the model:
```bash
python train.py
```

### Testing

To evaluate the model:
```bash
python test.py
```

## Model Architecture

The implementation follows the architecture described in several key papers:

1. **GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding**
   - Lepikhin, D., et al. (2020)
   - Key features: Capacity factor, load balancing

2. **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity**
   - Fedus, W., et al. (2021)
   - Key features: Simplified routing, improved training stability

3. **RoFormer: Enhanced Transformer with Rotary Position Embedding**
   - Su, J., et al. (2021)
   - Key features: Rotary Positional Embeddings

## Features

- **Efficient Routing**: Implements top-k routing with capacity factor
- **Load Balancing**: Uses auxiliary loss and router noise for balanced expert utilization
- **Memory Efficiency**: Gradient checkpointing for training large models
- **Modern Architecture**: Incorporates SwiGLU activation and RMSNorm
- **Flexible Configuration**: Easily configurable through `config.py`

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{lepikhin2020gshard,
  title={GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding},
  author={Lepikhin, Dmitry and Lee, HyoukJoong and Xu, Yuanzhong and Chen, Dehao and Firat, Orhan and Huang, Yanping and Krikun, Maxim and Shazeer, Noam and Chen, Zhifeng},
  journal={arXiv preprint arXiv:2006.16668},
  year={2020}
}

@article{fedus2021switch,
  title={Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity},
  author={Fedus, William and Zoph, Barret and Shazeer, Noam},
  journal={arXiv preprint arXiv:2101.03961},
  year={2021}
}
```

