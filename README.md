# Deep Learning Language Models Implementation

This repository contains implementations of advanced language models.

## Project Structure

```
.
├── MyMoe
│   ├── MoE.py
│   ├── Attention.py
│   ├── Transformer_Block.py
│   ├── ROPE.py
│   ├── RMSNorm.py
│   ├── training_config.py
│   ├── train.py
│   ├── test.py
│   ├── BpeTokenizer.py
│   ├── config.py
│   ├── constants.py
│   ├── archi.py
│   ├── input.txt
│   ├── smallinput.txt
│   └── tinyshakespeare_bpe_tokenizer.json
├── gpt2
│   └── train.py
├── runs
├── example.txt
└── .gitignore
```

## Mixture of Experts (MoE) Implementation

The MoE implementation is a modern, efficient implementation of the Mixture of Experts architecture, inspired by recent advances in large language models. It features:

### Key Features

- **Multi-Head Latent Attention (MLA)**: Advanced attention mechanism with latent vectors
  - Two-stage attention process:
    1. Latent vectors attend to context
    2. Query sequence attends to latent vectors
  - Configurable number of latent vectors (default: 16)
  - Efficient memory usage through latent compression
  - KV-cache support for faster inference
  - RoPE integration for better position awareness

- **Sparse MoE Layers**: Efficient implementation of sparse mixture of experts
- **Modern Architecture Components**:
  - SwiGLU activation
  - Rotary Positional Embeddings (RoPE)
  - RMSNorm layer normalization
  - Gradient checkpointing
- **Efficient Training**:
  - Mixed Precision Training with BF16
  - Top-k routing mechanism
  - Load balancing through auxiliary loss
  - Capacity factor for expert utilization
  - Router noise for improved training stability

### Model Architecture

- **Base Configuration**:
  - 12 transformer layers
  - 768 embedding dimension
  - 4 attention heads
  - 2 KV heads
  - 16 latent vectors for MLA
  - 2 experts per MoE layer
  - 128 maximum sequence length

### Implementation Details

The MoE implementation follows the architecture described in several key papers:

1. **GShard** (Lepikhin et al., 2020)
   - Capacity factor implementation
   - Load balancing techniques

2. **Switch Transformers** (Fedus et al., 2021)
   - Simplified routing mechanism
   - Improved training stability

3. **RoFormer** (Su et al., 2021)
   - Rotary Positional Embeddings

4. **Multi-Head Latent Attention**
   - Efficient attention through latent vectors
   - Two-stage attention mechanism
   - Memory-efficient implementation
   - KV-cache support for inference

For detailed implementation information, see the [MyMoe README](MyMoe/README.md).

## GPT-2 Implementation

The repository also includes a GPT-2 implementation, providing a baseline transformer architecture for comparison and experimentation.

### Features

- Standard GPT-2 architecture
- Training utilities
- Checkpoint management

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- NVIDIA A100 or similar for BF16 support

### Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training

#### MoE Model
```bash
cd MyMoe
python train.py
```

The training script automatically uses mixed precision training with BF16 when available on the hardware.

#### GPT-2 Model
```bash
cd gpt2
python train.py
```


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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## Work In Progress

- my one of main goal is to train the minideep on fineweb edu no matter the time on my rtx3060 gpu lmao but i have my finals for now so i will do come back to it 
- another goal is to implement a visualization mechanism like mav from attentionmech but he did it for gpt type models so im hoping i can do this for moe models 
- also the test.py is not for full moe only some files i uhh will update later  
