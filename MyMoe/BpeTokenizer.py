from typing import Optional 
from tokenizers import Tokenizer 
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import NFD,Lowercase,StripAccents,Sequence as NormalizerSequence #text normalization
from tokenizers.processors import TemplateProcessing
from constants import INPUT_DATA_FILE,TOKENIZER_FILE,CHECKPOINT_DIR,RUNS_DIR_BASE 
import os

from config import ModelArgs 

# Constants for the BPE tokenizer 
UNK_TOKEN = "[UNK]"
EOS_TOKEN = "[EOS]"
BOS_TOKEN = "[BOS]"
PAD_TOKEN = "[PAD]"
SPECIAL_TOKENS = [UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN]

#placeholders for the token ids which will be set up after training
# These will be set after training the tokenizer

POS_TOKEN_ID :Optional[int] = None 
EOS_TOKEN_ID :Optional[int] = None 
BOS_TOKEN_ID :Optional[int] = None 

def get_bpe_tokenizer(args_config:ModelArgs  , text_path : str,retrain : False):
    """
    function to train a BPE tokenizer and configure the model args.

    Args:
        args_config (ModelArgs): model args file
        text_path (str): dataset path
        retrain (False): Loads an existing BPE tokenizer or trains a new one
        if not found or `force_retrain` is True.


    Returns:
        Tuple[Tokenizer,ModelArgs]: updated tokenizer and modelargs (in model args only the vocab size is updated)
    """
    global PAD_TOKEN_ID, EOS_TOKEN_ID, BOS_TOKEN_ID # Allow modification of global token ID variables.

    if os.path.exists(TOKENIZER_FILE) and not retrain:
        tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
        print(f"Loaded existing BPE tokenizer from {TOKENIZER_FILE}")
    else:
        print(f"Training new BPE tokenizer from {text_path} with vocab size {args_config.vocab_size}")

        tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
        
        # normalizer for reducing the vocab size and standardizing the text
        tokenizer.normalizer = NormalizerSequence([
            NFD(),
            Lowercase(),
            StripAccents()
        ])
        # pre tokenization
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
        
        #initializing the BPE trainer 
        # here we set min frequency to 2 because we want the tokenizer 
        # to merge subword pairs that occur atleast 2 times 
        # so this way it wont merge any rare subwords which increase overall vocab size and computation
        trainer = BpeTrainer(
            vocab_size=args_config.vocab_size,
            special_tokens=SPECIAL_TOKENS,
            min_frequency=2, # Minimum frequency for subword units
            show_progress=True,
        )
        
        # Training weewoo weewoo
        print("Training the tokenizer")
        tokenizer.train([text_path], trainer=trainer)
        tokenizer.save(TOKENIZER_FILE)
        print(f"Tokenizer trained and saved to {TOKENIZER_FILE}")

       # retrieving the ids for the special tokens
        bos_id=tokenizer.token_to_id(BOS_TOKEN)
        eos_id=tokenizer.token_to_id(EOS_TOKEN)
        pad_id=tokenizer.token_to_id(PAD_TOKEN)

        if bos_id is None or eos_id is None or pad_id is None:
            raise ValueError("Tokenizer did not assign IDs to special tokens. Check the training process.")
        

        
        #post processor for adding special tokens
        # '$A' represents a input sequence
        # so the sequence becomes '[BOS] $A [EOS]'
        tokenizer.post_process=TemplateProcessing(
            single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
            special_tokens=[
                (BOS_TOKEN, bos_id),
                (EOS_TOKEN, eos_id),
            ]
        ) #we aren't using the unk token in this because we already have a special id for it
        
        tokenizer.enable_padding(pad_id=pad_id, pad_token=PAD_TOKEN, length=args_config.max_seq_len)
        tokenizer.enable_truncation(max_length=args_config.max_seq_len)
        
        PAD_TOKEN_ID = pad_id
        EOS_TOKEN_ID = eos_id
        BOS_TOKEN_ID = bos_id
        
    # Update the model args with the new vocab size
    actual_vocab_size = tokenizer.get_vocab_size()
    args_config = args_config._replace(vocab_size=actual_vocab_size)
    print(f"Updated ModelArgs with vocab size: {actual_vocab_size}")
    print(f"PAD_TOKEN_ID: {PAD_TOKEN_ID}, EOS_TOKEN_ID: {EOS_TOKEN_ID}, BOS_TOKEN_ID: {BOS_TOKEN_ID}")
   
    return tokenizer, args_config
