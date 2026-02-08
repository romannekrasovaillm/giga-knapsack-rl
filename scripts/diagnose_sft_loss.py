#!/usr/bin/env python3
"""
Diagnostic: inspect per-token loss distribution in SFT dataset.

Shows exactly which tokens contribute to loss and how much,
to understand why initial loss is ~0.7 instead of expected ~2.5-3.5.

Usage:
  python scripts/diagnose_sft_loss.py --model-name ai-sage/GigaChat3-10B-A1.8B-base \
    --data-cache-dir ./data/raw --max-samples 100 --num-inspect 5
"""

import argparse
import json
import logging
import sys

sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data.loader import NemotronAgenticLoader
from src.data.sft_dataset import SFTToolCallingDataset
from src.utils import get_attn_implementation, get_torch_dtype

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def diagnose(args):
    print(f"\n{'='*80}")
    print(f"{BOLD}SFT Loss Diagnostic{RESET}")
    print(f"{'='*80}\n")

    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=get_torch_dtype(),
        trust_remote_code=True,
        attn_implementation=get_attn_implementation(),
    )
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    # Load data
    print(f"Loading dataset (max_samples={args.max_samples})...")
    loader = NemotronAgenticLoader(cache_dir=args.data_cache_dir)
    records = loader.load_tool_calling(max_samples=args.max_samples)
    dataset = SFTToolCallingDataset(
        records=records, tokenizer=tokenizer, max_length=args.max_length,
    )

    # === Global stats ===
    all_losses = []
    all_loss_token_counts = []
    per_sample_avg_losses = []

    n_analyze = min(args.max_samples, len(dataset))
    print(f"\nAnalyzing {n_analyze} samples...\n")

    for idx in range(n_analyze):
        sample = dataset[idx]
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
        labels = sample["labels"].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Per-token CE loss (no reduction)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Flatten for cross_entropy
        vocab_size = shift_logits.shape[-1]
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)

        per_token_loss = F.cross_entropy(
            flat_logits, flat_labels, ignore_index=-100, reduction="none"
        )

        # Only non-ignored positions
        valid_mask = (flat_labels != -100)
        valid_losses = per_token_loss[valid_mask]

        if len(valid_losses) == 0:
            continue

        n_loss_tokens = len(valid_losses)
        avg_loss = valid_losses.mean().item()

        all_losses.append(valid_losses.cpu())
        all_loss_token_counts.append(n_loss_tokens)
        per_sample_avg_losses.append(avg_loss)

        # === Detailed inspection for first N samples ===
        if idx < args.num_inspect:
            print(f"\n{'='*80}")
            print(f"{BOLD}Sample {idx}{RESET} | "
                  f"loss_tokens={n_loss_tokens} | avg_loss={avg_loss:.4f} | "
                  f"perplexity={torch.exp(torch.tensor(avg_loss)):.1f}")
            print(f"{'='*80}")

            # Show tokens with their loss values
            seq_len = shift_labels.shape[1]
            tokens_shown = 0
            loss_idx = 0

            # Collect (token_str, loss, is_loss_token) tuples
            token_infos = []
            for pos in range(seq_len):
                label = shift_labels[0, pos].item()
                token_id = input_ids[0, pos + 1].item()
                token_str = tokenizer.decode([token_id])

                if label == -100:
                    token_infos.append((token_str, None, False))
                else:
                    loss_val = per_token_loss[pos].item()
                    token_infos.append((token_str, loss_val, True))

            # Print loss token distribution
            loss_values = [l for _, l, is_loss in token_infos if is_loss and l is not None]
            if loss_values:
                loss_t = torch.tensor(loss_values)
                bins = [0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, float('inf')]
                bin_names = ["<0.1", "0.1-0.5", "0.5-1.0", "1.0-2.0", "2.0-3.0", "3.0-5.0", ">5.0"]
                print(f"\n  {BOLD}Loss distribution:{RESET}")
                for i in range(len(bins) - 1):
                    count = ((loss_t >= bins[i]) & (loss_t < bins[i+1])).sum().item()
                    pct = count / len(loss_values) * 100
                    bar = "#" * int(pct / 2)
                    color = GREEN if bins[i] < 0.5 else (YELLOW if bins[i] < 2.0 else RED)
                    print(f"    {color}{bin_names[i]:>8s}: {count:>4d} ({pct:5.1f}%) {bar}{RESET}")

            # Print actual tokens (first ~200 loss tokens)
            print(f"\n  {BOLD}Tokens (loss tokens highlighted):{RESET}")
            printed_loss = 0
            line = "  "
            in_loss_region = False

            for token_str, loss_val, is_loss in token_infos:
                if printed_loss >= 200 and is_loss:
                    break

                # Clean token for display
                display = token_str.replace("\n", "\\n").replace("\t", "\\t")
                if len(display) > 20:
                    display = display[:17] + "..."

                if is_loss and loss_val is not None:
                    printed_loss += 1
                    if loss_val < 0.1:
                        color = GREEN  # trivially easy
                    elif loss_val < 1.0:
                        color = YELLOW  # easy
                    elif loss_val < 3.0:
                        color = ""  # moderate
                    else:
                        color = RED  # hard
                    reset = RESET if color else ""
                    chunk = f"{color}[{display}|{loss_val:.2f}]{reset}"
                    if not in_loss_region:
                        line += f"\n  {CYAN}>>>{RESET} "
                        in_loss_region = True
                else:
                    if in_loss_region:
                        in_loss_region = False
                    continue  # skip non-loss tokens in output

                line += chunk
                if len(line) > 200:
                    print(line)
                    line = "      "

            if line.strip():
                print(line)

        if (idx + 1) % 50 == 0:
            running_avg = sum(per_sample_avg_losses) / len(per_sample_avg_losses)
            print(f"  ... analyzed {idx+1}/{n_analyze}, running avg loss={running_avg:.4f}", flush=True)

    # === Summary ===
    print(f"\n\n{'='*80}")
    print(f"{BOLD}SUMMARY ({len(per_sample_avg_losses)} samples){RESET}")
    print(f"{'='*80}")

    avg = sum(per_sample_avg_losses) / len(per_sample_avg_losses)
    per_sample_t = torch.tensor(per_sample_avg_losses)
    all_flat = torch.cat(all_losses)

    print(f"\n  Per-sample avg loss: mean={avg:.4f}, median={per_sample_t.median():.4f}, "
          f"std={per_sample_t.std():.4f}")
    print(f"  Per-sample avg loss: min={per_sample_t.min():.4f}, max={per_sample_t.max():.4f}")
    print(f"  Overall perplexity: {torch.exp(torch.tensor(avg)):.2f}")

    print(f"\n  Per-token loss (all {len(all_flat):,} tokens):")
    print(f"    mean={all_flat.mean():.4f}, median={all_flat.median():.4f}, "
          f"std={all_flat.std():.4f}")
    print(f"    min={all_flat.min():.4f}, max={all_flat.max():.4f}")

    # Token loss histogram
    bins = [0, 0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, float('inf')]
    bin_names = ["<0.01", "0.01-0.1", "0.1-0.5", "0.5-1.0", "1.0-2.0",
                 "2.0-3.0", "3.0-5.0", "5.0-10", ">10"]
    print(f"\n  {BOLD}Global loss histogram:{RESET}")
    for i in range(len(bins) - 1):
        count = ((all_flat >= bins[i]) & (all_flat < bins[i+1])).sum().item()
        pct = count / len(all_flat) * 100
        bar = "#" * int(pct / 2)
        color = GREEN if bins[i] < 0.5 else (YELLOW if bins[i] < 2.0 else RED)
        print(f"    {color}{bin_names[i]:>10s}: {count:>7,d} ({pct:5.1f}%) {bar}{RESET}")

    # Percentage of "free" tokens (loss < 0.1)
    n_free = (all_flat < 0.1).sum().item()
    n_easy = (all_flat < 0.5).sum().item()
    n_hard = (all_flat >= 2.0).sum().item()
    print(f"\n  {GREEN}Free tokens (loss < 0.1): {n_free:,} ({n_free/len(all_flat)*100:.1f}%){RESET}")
    print(f"  {YELLOW}Easy tokens (loss < 0.5): {n_easy:,} ({n_easy/len(all_flat)*100:.1f}%){RESET}")
    print(f"  {RED}Hard tokens (loss >= 2.0): {n_hard:,} ({n_hard/len(all_flat)*100:.1f}%){RESET}")

    avg_loss_tokens = sum(all_loss_token_counts) / len(all_loss_token_counts)
    print(f"\n  Avg loss tokens per sample: {avg_loss_tokens:.0f}")
    print(f"  Total loss tokens analyzed: {len(all_flat):,}")

    # Recommendation
    print(f"\n{'='*80}")
    print(f"{BOLD}DIAGNOSIS:{RESET}")
    if n_free / len(all_flat) > 0.5:
        print(f"  {RED}>>> >50% of loss tokens are 'free' (loss<0.1).{RESET}")
        print(f"  The model already predicts most assistant content well.")
        print(f"  This is likely because tool-call JSON structure is predictable.")
        print(f"  Consider: the model may already know tool-calling from pre-training.")
    elif avg < 1.0:
        print(f"  {YELLOW}>>> Avg loss is low ({avg:.2f}) but distribution is mixed.{RESET}")
        print(f"  Some tokens are hard (function names, arg values), most are easy (JSON syntax).")
        print(f"  SFT is working but learning signal is diluted by structural tokens.")
    else:
        print(f"  {GREEN}>>> Loss looks normal ({avg:.2f}). Model is genuinely learning.{RESET}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose SFT loss")
    parser.add_argument("--model-name", default="ai-sage/GigaChat3-10B-A1.8B-base")
    parser.add_argument("--data-cache-dir", default="./data/raw")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--num-inspect", type=int, default=5,
                        help="Number of samples to show per-token detail")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    diagnose(args)
