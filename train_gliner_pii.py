"""
Fine-tune GLiNER on Indian PII dataset (pii_dataset_v3.xlsx)
Compatible with gliner 0.2.26

Fixes applied vs previous versions:
  - No GLiNERDataset (removed in 0.2.x) → ListDataset + SpanDataCollator
  - tokenizer= → processing_class= (HF Transformers ≥ 4.46)
  - SafeCollator: injects ALL_LABELS as fallback negatives when a batch
    contains only no-pii/noise rows (ner=[]) → prevents reshape([8,-1,0]) crash
  - fp16=True when CUDA is available, False otherwise (no manual --device needed)
"""

import re
import os
import json
import random
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from typing import Any

# ── 1. Entity label normalisation ────────────────────────────────────────────
LABEL_MAP = {
    "full_name":       "full_name",
    "name":            "full_name",
    "phone_number":    "phone_number",
    "aadhaar_number":  "aadhaar_number",
    "pan_card":        "pan_card",
    "company_name":    "company_name",
    "postal_address":  "postal_address",
    "date_of_birth":   "date_of_birth",
    "bank_account":    "bank_account",
    "email_address":   "email_address",
    "passport_number": "passport_number",
    "voter_id":        "voter_id",
    "drivers_license": "drivers_license",
    "card_number":     "card_number",
}

ALL_LABELS = sorted(set(LABEL_MAP.values()))


def normalise_label(raw: str) -> str | None:
    return LABEL_MAP.get(raw.strip().lower())


# ── 2. (Query, Masked) → GLiNER sample ───────────────────────────────────────

def build_sample(query: str, masked: str) -> dict | None:
    if not isinstance(query, str) or not isinstance(masked, str):
        return None
    query  = query.strip()
    masked = masked.strip()

    placeholder_re = re.compile(r'\[([A-Za-z_]+)\]')
    placeholders   = list(placeholder_re.finditer(masked))

    tokens = query.split()
    if not tokens:
        return None

    token_starts, token_ends = [], []
    pos = 0
    for tok in tokens:
        idx = query.index(tok, pos)
        token_starts.append(idx)
        token_ends.append(idx + len(tok))
        pos = idx + len(tok)

    def char_to_tok(char_idx, side="start"):
        for i, (ts, te) in enumerate(zip(token_starts, token_ends)):
            if side == "start" and ts <= char_idx < te:
                return i
            if side == "end"   and ts < char_idx <= te:
                return i
        dists = [abs(ts - char_idx) for ts in token_starts]
        return dists.index(min(dists))

    if not placeholders:
        return {"tokenized_text": tokens, "ner": []}

    ner_spans = []
    q_pos, m_pos = 0, 0

    for i, ph in enumerate(placeholders):
        label = normalise_label(ph.group(1))
        if label is None:
            return None

        prefix_len = ph.start() - m_pos
        if prefix_len < 0:
            return None
        q_pos += prefix_len
        m_pos  = ph.end()

        next_ph    = placeholders[i + 1] if i + 1 < len(placeholders) else None
        after_text = masked[m_pos:next_ph.start()] if next_ph else masked[m_pos:]

        if after_text:
            idx = query.find(after_text, q_pos)
            if idx == -1:
                idx = query.find(after_text.strip(), q_pos)
                if idx == -1:
                    return None
            span_end = idx
        else:
            span_end = len(query)

        if not query[q_pos:span_end].strip():
            return None

        ner_spans.append((q_pos, span_end, label))
        q_pos = span_end

    ner_token_spans = []
    for cs, ce, lbl in ner_spans:
        t_start = char_to_tok(cs, "start")
        t_end   = char_to_tok(ce, "end")
        if t_start <= t_end:
            ner_token_spans.append([t_start, t_end, lbl])

    return {"tokenized_text": tokens, "ner": ner_token_spans}


# ── 3. Load dataset ───────────────────────────────────────────────────────────

def load_dataset(xlsx_path: str, max_rows: int | None = None) -> list[dict]:
    df = pd.read_excel(xlsx_path, nrows=max_rows)
    df = df[["Query", "Masked"]].dropna(subset=["Query", "Masked"])

    samples, skipped = [], 0
    for _, row in df.iterrows():
        s = build_sample(str(row["Query"]), str(row["Masked"]))
        if s is not None:
            samples.append(s)
        else:
            skipped += 1

    print(f"Loaded {len(samples)} samples  |  {skipped} skipped (alignment errors)")
    return samples


# ── 4. PyTorch Dataset wrapper ────────────────────────────────────────────────

class ListDataset(Dataset):
    def __init__(self, data: list[dict]):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


# ── 5. Safe collator ──────────────────────────────────────────────────────────
# Root cause of reshape([8, -1, 0]):
#   When every sample in a batch has ner=[], batch_generate_class_mappings
#   builds an empty types list → class_to_id={} → num_classes=0.
#   The model then tries reshape(batch, seq, 0) which crashes.
#
# Fix: subclass SpanDataCollator and inject ALL_LABELS as ner_negatives on
# every sample so there's always at least one class in the batch.

class SafeCollator:
    """
    Wraps SpanDataCollator. Before collating, stamps ner_negatives=ALL_LABELS
    on every sample so batches of pure negatives (no-pii/noise rows) never
    produce an empty class set.
    """
    def __init__(self, base_collator, all_labels: list[str]):
        self.base      = base_collator
        self.all_labels = all_labels

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        # Inject fallback negatives so num_classes is always ≥ 1
        patched = []
        for sample in batch:
            s = dict(sample)
            if not s.get("ner"):
                s["ner_negatives"] = self.all_labels
            patched.append(s)
        return self.base(patched)


# ── 6. Train/val split ────────────────────────────────────────────────────────

def split(samples, val_ratio=0.1, seed=42):
    random.seed(seed)
    data = samples.copy()
    random.shuffle(data)
    n_val = max(1, int(len(data) * val_ratio))
    return data[n_val:], data[:n_val]


# ── 7. Fine-tune ──────────────────────────────────────────────────────────────

def train(
    xlsx_path:   str        = "pii_dataset_3000.xlsx",
    base_model:  str        = "urchade/gliner_medium-v2.1",
    output_dir:  str        = "./gliner_pii_finetuned",
    num_epochs:  int        = 5,
    batch_size:  int        = 8,
    lr:          float      = 5e-5,
    val_ratio:   float      = 0.1,
    max_rows:    int | None = None,   # None = use full dataset
):
    from gliner import GLiNER
    from gliner.training import Trainer, TrainingArguments
    from gliner.data_processing.collator import SpanDataCollator

    use_cuda = torch.cuda.is_available()
    print(f"CUDA available: {use_cuda}" + (f"  →  {torch.cuda.get_device_name(0)}" if use_cuda else "  →  training on CPU"))

    print(f"Loading base model: {base_model}")
    model = GLiNER.from_pretrained(base_model)

    base_collator = SpanDataCollator(
        config         = model.config,
        data_processor = model.data_processor,
        prepare_labels = True,
    )
    collator = SafeCollator(base_collator, ALL_LABELS)

    print("Parsing dataset …")
    samples = load_dataset(xlsx_path, max_rows=max_rows)
    train_data, val_data = split(samples, val_ratio=val_ratio)
    print(f"Train: {len(train_data)}  |  Val: {len(val_data)}")

    Path("gliner_train_data.json").write_text(json.dumps(train_data[:20], indent=2))
    print("Saved gliner_train_data.json (first 20 samples for inspection)")

    training_args = TrainingArguments(
        output_dir                  = output_dir,
        num_train_epochs            = num_epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size,
        learning_rate               = lr,
        weight_decay                = 0.01,
        warmup_ratio                = 0.1,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        logging_steps               = 50,
        fp16                        = use_cuda,   # auto: True on GPU, False on CPU
        dataloader_num_workers      = 0,
        focal_loss_alpha            = -1,
        focal_loss_gamma            = 0,
        negatives                   = 1.0,
        masking                     = "global",
    )

    trainer = Trainer(
        model            = model,
        args             = training_args,
        train_dataset    = ListDataset(train_data),
        eval_dataset     = ListDataset(val_data),
        data_collator    = collator,
        processing_class = model.data_processor.transformer_tokenizer,
    )

    print("Training …")
    trainer.train()

    print(f"Saving to {output_dir}")
    model.save_pretrained(output_dir)
    print("Done.")


# ── 8. Quick inference test ───────────────────────────────────────────────────

def test(model_dir: str = "./gliner_pii_finetuned"):
    from gliner import GLiNER

    model = GLiNER.from_pretrained(model_dir)
    cases = [
        "My aadhaar is 1234 5678 9012 and pan is ABCDE1234F",
        "Contact me at ravi.sharma@gmail.com or 9876543210",
        "I Priya Verma am the proprietor of ABC Enterprises",
        "aadhaar 5868 2635 4341 pan GXZEN9431O phone 7947604856",
        "I wish to bring to your kind notice that the order is issued long back.",
    ]
    for text in cases:
        entities = model.predict_entities(text, ALL_LABELS, threshold=0.5)
        print(f"\nText : {text}")
        print(f"Found: {[(e['text'], e['label']) for e in entities]}")


# ── 9. CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Fine-tune GLiNER on Indian PII data")
    p.add_argument("--xlsx",        default="pii_dataset_v3.xlsx")
    p.add_argument("--base_model",  default="urchade/gliner_medium-v2.1")
    p.add_argument("--output_dir",  default="./gliner_pii_finetuned")
    p.add_argument("--epochs",      type=int,   default=5)
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--lr",          type=float, default=5e-5)
    p.add_argument("--max_rows",    type=int,   default=None,
                   help="Cap rows loaded from xlsx (e.g. 200 for a quick test run)")
    p.add_argument("--test_only",   action="store_true")
    args = p.parse_args()

    if args.test_only:
        test(args.output_dir)
    else:
        train(
            xlsx_path  = args.xlsx,
            base_model = args.base_model,
            output_dir = args.output_dir,
            num_epochs = args.epochs,
            batch_size = args.batch_size,
            lr         = args.lr,
            max_rows   = args.max_rows,
        )
        test(args.output_dir)