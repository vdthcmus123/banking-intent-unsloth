import os, json, yaml, torch
from difflib import get_close_matches
from unsloth import FastLanguageModel
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, classification_report


class IntentClassification:

    def __init__(self, model_path: str):
        """Load config file, tokenizer và model checkpoint."""
        with open(model_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.max_new_tokens = cfg.get("max_new_tokens", 20)

        # Đọc label map
        with open(cfg["label_map_path"], "r") as f:
            lmap = json.load(f)

        self.valid_labels = lmap.get("valid_labels", list(lmap["label2id"].keys()))
        self.system_prompt = lmap.get("system_prompt", "")
        # Build lookup: lowercase → original (để fuzzy match)
        self._lower2orig = {l.lower(): l for l in self.valid_labels}

        # Load model + LoRA adapter
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name     = cfg["lora_checkpoint"],
            max_seq_length = cfg["max_seq_length"],
            load_in_4bit   = cfg["load_in_4bit"],
            dtype          = None,
        )
        FastLanguageModel.for_inference(self.model)
        print(f" Model loaded | {len(self.valid_labels)} valid labels | inference mode ON")

    def _snap_to_valid(self, raw_pred: str) -> str:
        # Thử exact match trước
        if raw_pred in self.valid_labels:
            return raw_pred

        # Thử case-insensitive
        raw_lower = raw_pred.strip().lower()
        if raw_lower in self._lower2orig:
            return self._lower2orig[raw_lower]

        # Fuzzy match: tìm label gần nhất theo edit distance
        matches = get_close_matches(
            raw_lower,
            list(self._lower2orig.keys()),
            n=1,
            cutoff=0.4   # threshold: 40% similarity
        )
        if matches:
            return self._lower2orig[matches[0]]

        # Fallback: tìm label nào là substring của prediction hoặc ngược lại
        for l_lower, l_orig in self._lower2orig.items():
            if l_lower in raw_lower or raw_lower in l_lower:
                return l_orig

        # Không tìm được → trả về raw
        return raw_pred

    def __call__(self, message: str) -> str:
        """Nhận câu văn bản, trả về nhãn intent dự đoán."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": message},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize              = True,
            add_generation_prompt = True,
            return_tensors        = "pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids      = inputs,
                max_new_tokens = self.max_new_tokens,
                do_sample      = False,
                temperature    = 1.0,
                use_cache      = True,
            )

        new_tokens  = outputs[0][inputs.shape[-1]:]
        raw_pred    = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        final_label = self._snap_to_valid(raw_pred)
        return final_label


def run_demo(clf):
    """ Demo inference trên 8 câu mẫu."""
    demo_messages = [
        "I was charged twice for the same purchase",
        "How do I transfer money to another account?",
        "My card was declined at the ATM",
        "I want a refund for this transaction",
        "What are the fees for international wire transfers?",
        "My balance hasn't updated after I deposited a cheque",
        "I noticed an extra charge on my statement",
        "The wrong amount of cash was given by the ATM",
    ]

    print("\n=== Inference Demo ===")
    for msg in demo_messages:
        pred = clf(msg)
        print(f"  Input : {msg}")
        print(f"  Intent: {pred}\n")


def run_evaluation(clf, test_csv_path: str):
    """ Đánh giá trên toàn bộ test set."""
    df_test = pd.read_csv(test_csv_path)

    print(" Đang chạy inference trên test set...")
    y_true, y_pred, y_raw = [], [], []

    for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
        messages = [
            {"role": "system", "content": clf.system_prompt},
            {"role": "user",   "content": row["text"]},
        ]
        inputs = clf.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(clf.model.device)

        with torch.no_grad():
            outputs = clf.model.generate(
                input_ids=inputs, max_new_tokens=clf.max_new_tokens,
                do_sample=False, temperature=1.0, use_cache=True,
            )
        new_tokens = outputs[0][inputs.shape[-1]:]
        raw  = clf.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        pred = clf._snap_to_valid(raw)

        y_true.append(row["label_name"])
        y_pred.append(pred)
        y_raw.append(raw)

    final_acc  = accuracy_score(y_true, y_pred)
    n_mismatch = sum(1 for r, p in zip(y_raw, y_pred) if r != p)
    print(f"\n Final Test Accuracy  : {final_acc*100:.2f}%")
    print(f" Label snap (mismatch): {n_mismatch}/{len(y_pred)} ({n_mismatch/len(y_pred)*100:.1f}%)")
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, zero_division=0))


if __name__ == "__main__":
    import sys
    config_path  = sys.argv[1] if len(sys.argv) > 1 else "configs/inference.yaml"
    test_csv     = sys.argv[2] if len(sys.argv) > 2 else "sample_data/test.csv"

    clf = IntentClassification(config_path)
    run_demo(clf)
    run_evaluation(clf, test_csv)
