INPUT_DATA_FILE = "input.txt"  # Path to the raw text file for training the tokenizer and model.
TOKENIZER_FILE = "tinyshakespeare_bpe_tokenizer.json" # Path to save/load the trained BPE tokenizer.
CHECKPOINT_DIR = "checkpoints_bpe_mla_moe" # Directory to save model checkpoints.
BEST_MODEL_FILENAME = "best_model_val_loss.pt" # Filename for the best model checkpoint based on validation loss.
RUNS_DIR_BASE = "runs_bpe_mla_moe" # Base directory for TensorBoard log files.
