# run_all_tests.py
import torch
import torch.nn.functional as F
import os
from typing import Optional, Tuple, List # For type hints, not strictly necessary for script execution

# --- Import from your project's modules ---
from config import ModelArgs
from Transformer_Block import LLM # Changed from Transformer_Block to model, assuming LLM is in model.py
import BpeTokenizer # Import the module itself
from constants import INPUT_DATA_FILE, TOKENIZER_FILE # Import constants

# Helper function to create a dummy text file for tokenizer training if it doesn't exist
def ensure_dummy_input_file_for_tokenizer():
    if not os.path.exists(INPUT_DATA_FILE):
        print(f"Creating dummy input file for tokenizer: {INPUT_DATA_FILE}")
        with open(INPUT_DATA_FILE, "w", encoding="utf-8") as f:
            f.write("First Citizen:\nBefore we proceed any further, hear me speak.\n\n")
            f.write("All:\nSpeak, speak.\n\n")
            f.write("First Citizen:\nYou are all resolved rather to die than to famish?\n\n")
            f.write("All:\nResolved. resolved.\n\n")
            f.write("First Citizen:\nFirst, you know Caius Marcius is chief enemy to the people.\n\n")
            f.write("All:\nWe know't, we know't.\n\n")
            f.write("First Citizen:\nLet us kill him, and we'll have corn at our own price.\n")
            f.write("Is't a verdict?\n")
            f.write("All:\nNo more talking on't; let it be done: away, away!\n")
            f.write("Second Citizen:\nOne word, good citizens.\n")
            for i in range(50):
                 f.write(f"This is an example sentence number {i} for the bpe tokenizer.\n")
                 f.write(f"Another line for testing the tokenizer and the dataset, line number {i+50} perhaps.\n")
                 f.write("The quick brown fox jumps over the lazy dog.\n")
                 f.write("Pack my box with five dozen liquor jugs.\n")
        print(f"Dummy file {INPUT_DATA_FILE} created.")


def create_attention_mask_from_pad_id(input_ids: torch.Tensor, pad_id: Optional[int]) -> Optional[torch.Tensor]:
    if pad_id is None:
        return None
    mask = (input_ids != pad_id)
    return mask


# --- Main Test Function ---
def run_extensive_tests():
    print("--- Starting Extensive Model Tests ---")
    ensure_dummy_input_file_for_tokenizer()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args_base = ModelArgs(
        vocab_size=500,
        embed_dim=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        num_latents=8,
        moe_num_experts=2,
        moe_top_k=1,
        max_seq_len=64,
        dropout_rate=0.1,
        moe_dropout_rate=0.1,
        use_gradient_checkpointing=False,
        norm_eps=1e-5,
        rope_theta=10000.0
    )

    print("\n--- Test: Tokenizer Initialization ---")
    # Construct absolute path for TOKENIZER_FILE based on constants.py location
    # This assumes constants.py and TOKENIZER_FILE are correctly defined relative to project root
    # or TOKENIZER_FILE in constants.py is an absolute path or relative to where constants.py is.
    # For simplicity, if TOKENIZER_FILE is just a filename, we'll make it relative to script dir.
    tokenizer_file_path_in_test = TOKENIZER_FILE # Use path from constants
    if not os.path.isabs(tokenizer_file_path_in_test): # If it's not absolute
        # Make it relative to this script's directory IF constants.py defines it as a simple filename
        # This part might need adjustment based on how TOKENIZER_FILE is defined in constants.py
        # The BpeTokenizer.py itself now uses os.path.dirname(__file__) which is more robust.
        # We should rely on that. This check is more for the os.remove here.
        pass


    if os.path.exists(tokenizer_file_path_in_test):
        print(f"Removing existing tokenizer file: {tokenizer_file_path_in_test}")
        os.remove(tokenizer_file_path_in_test)

    # Call get_bpe_tokenizer from the imported BpeTokenizer module
    tokenizer, args_updated = BpeTokenizer.get_bpe_tokenizer(args_base, INPUT_DATA_FILE, retrain=True)
    assert args_updated.vocab_size == tokenizer.get_vocab_size(), "Vocab size mismatch"
    # Access PAD_TOKEN_ID via the BpeTokenizer module's namespace
    assert BpeTokenizer.PAD_TOKEN_ID is not None, "PAD_TOKEN_ID not set by tokenizer module"
    print(f"Tokenizer loaded. Vocab size: {args_updated.vocab_size}, PAD_ID: {BpeTokenizer.PAD_TOKEN_ID}")
    print(f"Actual ModelArgs used for tests: {args_updated}")


    print("\n--- Test: Dataset & Input Preparation ---")
    TEMP_DATA_FILE_FOR_MODEL = "temp_model_test_data.txt"
    test_sentences_for_model = [
        "This is a test sentence for our model, it should be reasonably long.",
        "Another example line to process with the tokenizer and feed into the LLM.",
        "Yet one more to ensure we have data, perhaps with some repeated words words.",
        "The quick brown fox jumps over the lazy dog always and forever, making it a full sentence.",
        "A much shorter one.",
        "Extremely long line to test truncation " + " ".join(["word"] * args_updated.max_seq_len),
    ] * (args_updated.max_seq_len // 4 + 1)

    with open(TEMP_DATA_FILE_FOR_MODEL, "w", encoding="utf-8") as f:
        for s in test_sentences_for_model:
            f.write(s + "\n")

    print(f"Using PAD_TOKEN_ID: {BpeTokenizer.PAD_TOKEN_ID} for direct tokenization.")

    batch_size = 4
    all_tokenized_sequences = []
    for i in range(batch_size):
        sentence = test_sentences_for_model[i % len(test_sentences_for_model)]
        encoding = tokenizer.encode(sentence)
        all_tokenized_sequences.append(torch.tensor(encoding.ids, dtype=torch.long))

    if not all_tokenized_sequences:
        raise ValueError("Failed to create any tokenized sequences for testing the model.")

    batch_full_ids = torch.stack(all_tokenized_sequences).to(device)
    assert batch_full_ids.shape == (batch_size, args_updated.max_seq_len), \
           f"Batch shape mismatch: {batch_full_ids.shape}"

    kv_context_split_len = args_updated.max_seq_len // 3
    query_split_len = args_updated.max_seq_len - kv_context_split_len

    batch_kv_context_ids = batch_full_ids[:, :kv_context_split_len].clone()
    batch_query_ids = batch_full_ids[:, kv_context_split_len:].clone()

    print(f"Batch Query IDs shape: {batch_query_ids.shape}")
    print(f"Batch KV Context IDs shape: {batch_kv_context_ids.shape}")

    kv_context_mask = create_attention_mask_from_pad_id(batch_kv_context_ids, BpeTokenizer.PAD_TOKEN_ID)
    if kv_context_mask is not None:
      print(f"KV Context Mask shape: {kv_context_mask.shape}, dtype: {kv_context_mask.dtype}")


    print("\n--- Test: Model Initialization & Basic Forward Pass (Train/Eval) ---")
    models_to_test_configs = {
        "GQA_default": args_updated,
        "MHA": args_updated._replace(num_kv_heads=args_updated.num_heads),
    }
    if args_updated.num_heads > 1 :
        models_to_test_configs["MQA"] = args_updated._replace(num_kv_heads=1)

    for model_name, current_args in models_to_test_configs.items():
        print(f"\n--- Testing Model Variant: {model_name} with Args: {current_args} ---")
        current_model = LLM(current_args).to(device)

        current_model.train()
        logits_train, cache_train, aux_loss_train = current_model(
            input_ids=batch_query_ids,
            kv_context_ids=batch_kv_context_ids,
            kv_context_mask=kv_context_mask,
            start_pos=0,
            use_cache=True
        )
        assert logits_train.shape == (batch_size, query_split_len, current_args.vocab_size), f"{model_name} train logits shape error"
        assert aux_loss_train is not None, f"{model_name} train aux_loss is None"
        if current_args.num_layers > 0 and cache_train:
             assert len(cache_train) == current_args.num_layers, f"{model_name} train cache length error"
             if cache_train[0] is not None:
                assert cache_train[0][0].shape == (batch_size, current_args.num_latents, current_args.embed_dim), f"{model_name} train cache K' shape error"
        print(f"{model_name} Train mode forward OK. Aux loss: {aux_loss_train.item():.4f}")

        current_model.eval()
        logits_eval, _, aux_loss_eval = current_model(
            input_ids=batch_query_ids,
            kv_context_ids=batch_kv_context_ids,
            kv_context_mask=kv_context_mask,
            start_pos=0,
            use_cache=False
        )
        assert logits_eval.shape == (batch_size, query_split_len, current_args.vocab_size), f"{model_name} eval logits shape error"
        print(f"{model_name} Eval mode forward OK. Aux loss: {aux_loss_eval.item():.4f}")

        if current_args.dropout_rate > 0 and not torch.allclose(logits_train, logits_eval, atol=1e-6) :
            print(f"{model_name} Train and Eval logits differ as expected due to dropout.")
        elif current_args.dropout_rate == 0 and not torch.allclose(logits_train, logits_eval, atol=1e-6):
            # Check if MoE router noise is a factor if it's enabled in ModelArgs
            moe_router_noise_active = hasattr(current_args, 'moe_router_noise') and current_args.moe_router_noise > 0
            if not moe_router_noise_active:
                 print(f"{model_name} WARNING: Train and Eval logits differ but dropout is 0 and MoE router noise seems off. Check other randomness.")
            else:
                 print(f"{model_name} Train and Eval logits differ, possibly due to MoE router noise in training mode.")


    model = LLM(args_updated).to(device)

    print("\n--- Test: Gradient Flow ---")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()

    logits_grad, _, aux_loss_grad = model(
        input_ids=batch_query_ids,
        kv_context_ids=batch_kv_context_ids,
        kv_context_mask=kv_context_mask,
        start_pos=0
    )
    dummy_target = torch.randint(0, args_updated.vocab_size, logits_grad.shape[:-1], device=device)
    loss = F.cross_entropy(logits_grad.reshape(-1, args_updated.vocab_size), dummy_target.reshape(-1)) + aux_loss_grad
    loss.backward()

    has_grads = any(param.grad is not None and param.grad.abs().sum() > 0 for param in model.parameters() if param.requires_grad)
    assert has_grads, "No gradients found or all gradients are zero on model parameters!"
    optimizer.step()
    print("Gradient flow and optimizer step OK.")


    print("\n--- Test: Detailed Latent KV Caching ---")
    model.eval()

    print("Cache Test - Pass 1 (Populate with initial query & kv_context)")
    _, cache_pass1, _ = model(
        input_ids=batch_query_ids,
        kv_context_ids=batch_kv_context_ids,
        kv_context_mask=kv_context_mask,
        start_pos=0,
        use_cache=True
    )
    assert cache_pass1 is not None and len(cache_pass1) == args_updated.num_layers, "Pass 1 cache error"
    if args_updated.num_layers > 0 and cache_pass1[0] is not None:
        print(f"Cache populated. Layer 0 K' shape: {cache_pass1[0][0].shape}")

    print("Cache Test - Pass 2 (Use cache, new query segment, same KV context)")
    next_query_segment_len = query_split_len // 2 if query_split_len > 1 else 1
    dummy_next_query_ids = torch.randint(0, args_updated.vocab_size,
                                        (batch_size, next_query_segment_len),
                                        device=device)
    
    logits_pass2a, cache_pass2a, _ = model(
        input_ids=dummy_next_query_ids,
        kv_context_ids=batch_kv_context_ids,
        kv_context_mask=kv_context_mask,
        start_pos=0,
        past_layer_latents_list=cache_pass1,
        use_cache=True
    )
    assert logits_pass2a.shape == (batch_size, next_query_segment_len, args_updated.vocab_size), "Pass 2a logits shape error"
    assert cache_pass2a is not None, "Pass 2a cache is None"
    if args_updated.num_layers > 0 and cache_pass1[0] is not None and cache_pass2a[0] is not None:
        assert cache_pass2a[0][0] is cache_pass1[0][0], "Pass 2a: Latent K' cache object NOT REUSED! Stage 1 recomputed."
        assert cache_pass2a[0][1] is cache_pass1[0][1], "Pass 2a: Latent V' cache object NOT REUSED! Stage 1 recomputed."
    print("Cache Test Pass 2a (new query, cached latents from same KV context) OK. Cache objects reused.")

    print("Cache Test - Pass 3 (Cache invalidation - new KV context, new latents)")
    new_kv_context_ids = torch.randint(0, args_updated.vocab_size, batch_kv_context_ids.shape, device=device)
    while torch.equal(new_kv_context_ids, batch_kv_context_ids):
        new_kv_context_ids = torch.randint(0, args_updated.vocab_size, batch_kv_context_ids.shape, device=device)
    new_kv_context_mask = create_attention_mask_from_pad_id(new_kv_context_ids, BpeTokenizer.PAD_TOKEN_ID) # Use module access

    _, cache_pass3_new_kv, _ = model(
        input_ids=batch_query_ids,
        kv_context_ids=new_kv_context_ids,
        kv_context_mask=new_kv_context_mask,
        start_pos=0,
        past_layer_latents_list=None,
        use_cache=True
    )
    assert cache_pass3_new_kv is not None, "Pass 3 new cache is None"
    if args_updated.num_layers > 0 and cache_pass1[0] is not None and cache_pass3_new_kv[0] is not None:
        assert cache_pass3_new_kv[0][0] is not cache_pass1[0][0], \
            "Pass 3: New Latent K' is SAME OBJECT as old K'. Cache not recomputed for new KV context."
        if not torch.allclose(cache_pass3_new_kv[0][0], cache_pass1[0][0]):
            print("Cache Test Pass 3 (new KV context) OK. Latent cache content differs from Pass 1.")
        else:
            print("Cache Test Pass 3 WARNING: New KV context latents are numerically identical to old ones. Could be coincidence or an issue if inputs were very different.")
    
    print("\n--- Test: Max Sequence Length RoPE Handling ---")
    long_query_ids = torch.randint(0, args_updated.vocab_size,
                                   (batch_size, args_updated.max_seq_len), device=device)
    short_kv_context_ids = batch_kv_context_ids[:, :args_updated.max_seq_len // 4].clone()
    short_kv_context_mask = create_attention_mask_from_pad_id(short_kv_context_ids, BpeTokenizer.PAD_TOKEN_ID) # Use module access
    try:
        _ = model(input_ids=long_query_ids, kv_context_ids=short_kv_context_ids, kv_context_mask=short_kv_context_mask, start_pos=0, use_cache=False)
        print("RoPE Test (query_len = max_seq_len) OK.")
    except IndexError as e:
        print(f"RoPE Test (query_len = max_seq_len) FAILED: {e}")
        raise
    
    short_query_ids_for_long_kv = batch_query_ids[:, :args_updated.max_seq_len // 4].clone()
    long_kv_context_ids = torch.randint(0, args_updated.vocab_size,
                                   (batch_size, args_updated.max_seq_len), device=device)
    long_kv_context_mask = create_attention_mask_from_pad_id(long_kv_context_ids, BpeTokenizer.PAD_TOKEN_ID) # Use module access
    try:
        _ = model(input_ids=short_query_ids_for_long_kv, kv_context_ids=long_kv_context_ids, kv_context_mask=long_kv_context_mask, start_pos=0, use_cache=False)
        print("RoPE Test (kv_context_len = max_seq_len) OK.")
    except IndexError as e:
        print(f"RoPE Test (kv_context_len = max_seq_len) FAILED: {e}")
        raise

    if args_updated.num_layers > 0:
        print("\n--- Test: Gradient Checkpointing ---")
        args_gc = args_updated._replace(use_gradient_checkpointing=True)
        model_gc = LLM(args_gc).to(device)
        model_gc.train()

        q_ids_gc = batch_query_ids.clone().detach()
        kv_ids_gc = batch_kv_context_ids.clone().detach()
        kv_mask_gc = kv_context_mask.clone().detach() if kv_context_mask is not None else None
        
        logits_gc, _, aux_loss_gc = model_gc(
            input_ids=q_ids_gc,
            kv_context_ids=kv_ids_gc,
            kv_context_mask=kv_mask_gc,
            start_pos=0
        )
        dummy_target_gc = torch.randint(0, args_gc.vocab_size, logits_gc.shape[:-1], device=device)
        loss_gc = F.cross_entropy(logits_gc.reshape(-1, args_gc.vocab_size), dummy_target_gc.reshape(-1)) + aux_loss_gc
        
        try:
            loss_gc.backward()
            has_grads_gc = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model_gc.parameters() if p.requires_grad)
            assert has_grads_gc, "No grads or all grads zero with gradient checkpointing!"
            print("Gradient checkpointing backward pass OK.")
        except Exception as e:
            print(f"Error during gradient checkpointing backward pass: {e}")
            raise
    else:
        print("\nSkipping Gradient Checkpointing test as num_layers is 0.")


    if os.path.exists(TEMP_DATA_FILE_FOR_MODEL):
        os.remove(TEMP_DATA_FILE_FOR_MODEL)

    print("\n--- All Extensive Tests Completed Successfully ---")

if __name__ == "__main__":
    run_extensive_tests()
