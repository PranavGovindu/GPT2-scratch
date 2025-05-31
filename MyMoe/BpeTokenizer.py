from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence as NormalizerSequence
from tokenizers.processors import TemplateProcessing
from constants import TOKENIZER_FILE
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


# Constants for the BPE tokenizer
UNK_TOKEN = "[UNK]"
EOS_TOKEN = "[EOS]"
BOS_TOKEN = "[BOS]"
PAD_TOKEN = "[PAD]"
SPECIAL_TOKENS = [UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN]


def get_bpe_tokenizer(args_config, text_path, retrain):
    tokenizer_file_abs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), TOKENIZER_FILE)

    if os.path.exists(tokenizer_file_abs_path) and not retrain:
        tokenizer = Tokenizer.from_file(tokenizer_file_abs_path)
        print(f"Loaded existing BPE tokenizer from {tokenizer_file_abs_path}")
    else:
        print(f"Training new BPE tokenizer from {text_path} with vocab size {args_config.vocab_size}")
        tokenizer = _train_new_tokenizer(text_path, args_config, tokenizer_file_abs_path)

    # Configure tokenizer with special tokens
    _configure_tokenizer(tokenizer, args_config)
    
    # Update vocab size and return
    actual_vocab_size = tokenizer.get_vocab_size()
    args_config.vocab_size = actual_vocab_size
    print(f"Updated ModelArgs with vocab size: {actual_vocab_size}")
    
    # Print token IDs for reference
    pad_id = tokenizer.token_to_id(PAD_TOKEN)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    bos_id = tokenizer.token_to_id(BOS_TOKEN)
    print(f"PAD_TOKEN_ID: {pad_id}, EOS_TOKEN_ID: {eos_id}, BOS_TOKEN_ID: {bos_id}")

    return tokenizer, args_config


def _train_new_tokenizer(text_path, args_config, save_path):
    """Train a new BPE tokenizer from scratch."""
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))

    tokenizer.normalizer = NormalizerSequence([
        NFD(),
        Lowercase(),
        StripAccents()
    ])
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)

    trainer = BpeTrainer(
        vocab_size=args_config.vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )

    print("Starting the training process")
    tokenizer.train([text_path], trainer=trainer)
    tokenizer.save(save_path)
    print(f"Tokenizer trained and saved to {save_path}")

    return tokenizer


def _configure_tokenizer(tokenizer, args_config):
    """Configure tokenizer with post-processing, padding, and truncation."""
    # Get special token IDs
    bos_id = tokenizer.token_to_id(BOS_TOKEN)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    pad_id = tokenizer.token_to_id(PAD_TOKEN)

    # Validate special tokens exist
    if any(token_id is None for token_id in [bos_id, eos_id, pad_id]):
        missing_tokens = []
        if bos_id is None:
            missing_tokens.append(BOS_TOKEN)
        if eos_id is None:
            missing_tokens.append(EOS_TOKEN)
        if pad_id is None:
            missing_tokens.append(PAD_TOKEN)
        raise ValueError(f"Missing special token IDs: {missing_tokens}")

    # Configure post-processing
    tokenizer.post_processor = TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        special_tokens=[
            (BOS_TOKEN, bos_id),
            (EOS_TOKEN, eos_id),
        ]
    )

    # Enable padding and truncation
    tokenizer.enable_padding(pad_id=pad_id, pad_token=PAD_TOKEN, length=args_config.max_seq_len)
    tokenizer.enable_truncation(max_length=args_config.max_seq_len)


class BPEDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples = []

        print(f"Processing dataset from {file_path}")
        self._load_and_tokenize(file_path)

        if not self.examples:
            print(f"No valid examples created from {file_path}")
        else:
            print(f"Created {len(self.examples)} examples from {file_path}")

    def _load_and_tokenize(self, file_path):
        """Load text file and create tokenized examples."""
        with open(file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()

        lines = full_text.splitlines()
        for line in tqdm(lines, desc="Tokenizing lines for BPEDataset"):
            if not line.strip():
                continue

            encoding = self.tokenizer.encode(line)
            token_ids = encoding.ids
            attention_mask_ids = encoding.attention_mask

            # Need at least 2 tokens to create input-target pairs
            if len(token_ids) < 2:
                continue

            # Create input-target pairs for next token prediction
            input_sequence = torch.tensor(token_ids[:-1], dtype=torch.long)
            target_sequence = torch.tensor(token_ids[1:], dtype=torch.long)
            input_attention_mask = torch.tensor(attention_mask_ids[:-1], dtype=torch.long)

            self.examples.append({
                "input_ids": input_sequence,
                "target_ids": target_sequence,
                "attention_mask": input_attention_mask
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
