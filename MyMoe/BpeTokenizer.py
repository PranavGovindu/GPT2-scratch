
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import NFD,Lowercase,StripAccents,Sequence as NormalizerSequence
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

# define golabal variables for token ids
PAD_TOKEN_ID = None
EOS_TOKEN_ID = None
BOS_TOKEN_ID = None

def get_bpe_tokenizer(args_config, text_path, retrain):
    global PAD_TOKEN_ID, EOS_TOKEN_ID, BOS_TOKEN_ID

    tokenizer_file_abs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), TOKENIZER_FILE)

    if os.path.exists(tokenizer_file_abs_path) and not retrain:
        tokenizer = Tokenizer.from_file(tokenizer_file_abs_path)
        print(f"Loaded existing BPE tokenizer from {tokenizer_file_abs_path}")

        pad_id = tokenizer.token_to_id(PAD_TOKEN)
        eos_id = tokenizer.token_to_id(EOS_TOKEN)
        bos_id = tokenizer.token_to_id(BOS_TOKEN)

        if pad_id is None or eos_id is None or bos_id is None:
            print(f"Warning: Loaded tokenizer from {tokenizer_file_abs_path} missing one or more special token IDs.")
        
        PAD_TOKEN_ID = pad_id
        EOS_TOKEN_ID = eos_id
        BOS_TOKEN_ID = bos_id

        if BOS_TOKEN_ID is not None and EOS_TOKEN_ID is not None:
            tokenizer.post_processor=TemplateProcessing(
                single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
                special_tokens=[
                    (BOS_TOKEN, BOS_TOKEN_ID),
                    (EOS_TOKEN, EOS_TOKEN_ID),
                ]
            )
        if PAD_TOKEN_ID is not None:
            tokenizer.enable_padding(pad_id=PAD_TOKEN_ID, pad_token=PAD_TOKEN, length=args_config.max_seq_len)
        tokenizer.enable_truncation(max_length=args_config.max_seq_len)

    else:
        print(f"Training new BPE tokenizer from {text_path} with vocab size {args_config.vocab_size}")

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

        print("starting the training process")
        tokenizer.train([text_path], trainer=trainer)
        tokenizer.save(tokenizer_file_abs_path)
        print(f"Tokenizer trained and saved to {tokenizer_file_abs_path}")

        bos_id=tokenizer.token_to_id(BOS_TOKEN)
        eos_id=tokenizer.token_to_id(EOS_TOKEN)
        pad_id=tokenizer.token_to_id(PAD_TOKEN)

        if bos_id is None or eos_id is None or pad_id is None:
            raise ValueError("Tokenizer did not assign IDs to special tokens after training. Check the training process.")

        tokenizer.post_processor=TemplateProcessing(
            single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
            special_tokens=[
                (BOS_TOKEN, bos_id),
                (EOS_TOKEN, eos_id),
            ]
        )
        tokenizer.enable_padding(pad_id=pad_id, pad_token=PAD_TOKEN, length=args_config.max_seq_len)
        tokenizer.enable_truncation(max_length=args_config.max_seq_len)

        PAD_TOKEN_ID = pad_id
        EOS_TOKEN_ID = eos_id
        BOS_TOKEN_ID = bos_id

    final_pad_id_check = tokenizer.token_to_id(PAD_TOKEN)
    final_eos_id_check = tokenizer.token_to_id(EOS_TOKEN)
    final_bos_id_check = tokenizer.token_to_id(BOS_TOKEN)

    if final_pad_id_check is None:
        print(f"Warning: PAD_TOKEN ('{PAD_TOKEN}') ID not found in the final tokenizer instance.")
    else:
        PAD_TOKEN_ID = final_pad_id_check

    if final_eos_id_check is None:
        print(f"Warning: EOS_TOKEN ('{EOS_TOKEN}') ID not found in the final tokenizer instance.")
    else:
        EOS_TOKEN_ID = final_eos_id_check
    
    if final_bos_id_check is None:
        print(f"Warning: BOS_TOKEN ('{BOS_TOKEN}') ID not found in the final tokenizer instance.")
    else:
        BOS_TOKEN_ID = final_bos_id_check

    actual_vocab_size = tokenizer.get_vocab_size()
    args_config.vocab_size = actual_vocab_size
    print(f"Updated ModelArgs with vocab size: {actual_vocab_size}")
    print(f"PAD_TOKEN_ID: {PAD_TOKEN_ID}, EOS_TOKEN_ID: {EOS_TOKEN_ID}, BOS_TOKEN_ID: {BOS_TOKEN_ID}")

    return tokenizer, args_config


#dataset class for BPE
class BPEDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples = []

        print(f"Processing dataset from {file_path} ")
        with open(file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()

        lines = full_text.splitlines()
        for line in tqdm(lines, desc="Tokenizing lines for BPEDataset"):
            if not line.strip():
                continue

            encoding = self.tokenizer.encode(line) 
            token_ids = encoding.ids
                
                
            # extract attention mask, ensuring it matches the input sequence length

            attention_mask_ids = encoding.attention_mask 

            if len(token_ids) >= 2: # Ensure there's at least one token to predict
                input_sequence = torch.tensor(token_ids[:-1], dtype=torch.long)
                target_sequence = torch.tensor(token_ids[1:], dtype=torch.long)
                
                # Ensure input_sequence and target_sequence are of the same length
                input_attention_mask = torch.tensor(attention_mask_ids[:-1], dtype=torch.long)

                self.examples.append({
                    "input_ids": input_sequence, 
                    "target_ids": target_sequence,
                    "attention_mask": input_attention_mask 
                })

        if not self.examples:
            print(f"no valid pairs from the dataset {file_path}")

        print(f"Created {len(self.examples)} pairs from {file_path}.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
