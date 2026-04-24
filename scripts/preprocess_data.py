import os, json, re, yaml, random
import numpy as np
import pandas as pd
from datasets import load_dataset


def main():
    with open("configs/train.yaml", "r") as f:
        CFG = yaml.safe_load(f)

    os.makedirs(CFG["sample_data_dir"], exist_ok=True)
    os.makedirs(CFG["output_dir"],      exist_ok=True)

    # ── Tải Dataset ──────────────────────────────────────────────────
    raw_ds = load_dataset("PolyAI/banking77", trust_remote_code=True)
    label_names = raw_ds["train"].features["label"].names

    print(f"Train: {len(raw_ds['train'])} | Test: {len(raw_ds['test'])}")
    print(f"Tổng số intent: {len(label_names)}")

    df_raw = pd.DataFrame(raw_ds["train"])
    df_raw["label_name"] = df_raw["label"].map(lambda x: label_names[x])
    print(df_raw[["text", "label_name"]].head(5).to_string())

    # ── EDA Phân phối Intent ─────────────────────────────────────────
    intent_counts = df_raw["label_name"].value_counts()
    print(f"Số lượng mẫu mỗi intent (min={intent_counts.min()}, max={intent_counts.max()}, mean={intent_counts.mean():.1f})")
    print(f"\nTop 25 intent nhiều mẫu nhất:")
    print(intent_counts.head(25).to_string())

    # ── Subset, Normalize, Data Augmentation, Few-shot Prompt ────────
    random.seed(42)
    np.random.seed(42)

    # 1. Chọn 20 intent
    all_counts = df_raw["label_name"].value_counts()
    top_40 = list(all_counts.head(40).index)

    blacklist = {
        "activate_card",
        "Refund_not_showing",
        "transfer_not_received",
        "transfer_fee",
        "declined_debit_or_credit_card_payment",
    }
    top_intents = [i for i in top_40 if i not in blacklist][:CFG["num_subset_intents"]]
    print(f"\n Đã chọn {len(top_intents)} intent:")
    for i, name in enumerate(top_intents, 1):
        print(f"  {i:2d}. {name}")

    # 2. Normalize text
    def normalize_text(t: str) -> str:
        t = t.strip()
        t = re.sub(r"\s+", " ", t)
        return t

    # 3. Data Augmentation
    PREFIXES = [
        "I need help because ", "Can you assist me? ",
        "Hello, I have an issue: ", "Hi there! ",
        "I am writing to report that ", "I'm contacting support because ",
        "I've noticed that ", "I need to inform you that ",
    ]
    SUFFIXES = [
        " Please help me.", " What should I do?",
        " Can you fix this issue?", " I need this resolved urgently.",
        " Please advise.", " I'd appreciate your help.",
    ]
    REPLACEMENTS = {
        "card"    : ["debit card", "credit card", "bank card"],
        "money"   : ["funds", "cash", "amount"],
        "transfer": ["send money", "wire transfer", "bank transfer"],
        "charge"  : ["fee", "cost", "deduction"],
        "account" : ["bank account", "my account"],
    }

    def augment_text(text: str) -> list:
        """Tạo các biến thể của text bằng các phép augmentation đơn giản."""
        variants = []
        t = text.strip()

        # Augment 1: Prefix
        if random.random() < CFG["aug_rate"]:
            variants.append(random.choice(PREFIXES) + t[0].lower() + t[1:])

        # Augment 2: Suffix
        if random.random() < CFG["aug_rate"]:
            variants.append(t + random.choice(SUFFIXES))

        # Augment 3: Lowercase toàn bộ
        if random.random() < CFG["aug_rate"]:
            variants.append(t.lower())

        # Augment 4: Thay synonym một từ ngẫu nhiên
        if random.random() < CFG["aug_rate"]:
            words = t.split()
            new_words = []
            changed = False
            for w in words:
                w_lower = w.lower().rstrip(".,?!")
                if w_lower in REPLACEMENTS and not changed:
                    new_words.append(random.choice(REPLACEMENTS[w_lower]))
                    changed = True
                else:
                    new_words.append(w)
            if changed:
                variants.append(" ".join(new_words))

        return variants

    # 4. Lọc & augment training data
    df_train_raw = pd.DataFrame(raw_ds["train"])
    df_train_raw["label_name"] = df_train_raw["label"].map(lambda x: label_names[x])
    df_train_raw["text"]       = df_train_raw["text"].map(normalize_text)

    df_test_raw = pd.DataFrame(raw_ds["test"])
    df_test_raw["label_name"] = df_test_raw["label"].map(lambda x: label_names[x])
    df_test_raw["text"]       = df_test_raw["text"].map(normalize_text)

    df_train_sub = df_train_raw[df_train_raw["label_name"].isin(top_intents)].copy()
    df_test_sub  = df_test_raw[df_test_raw["label_name"].isin(top_intents)].copy()

    aug_rows = []
    for _, row in df_train_sub.iterrows():
        variants = augment_text(row["text"])
        for v in variants:
            aug_rows.append({"text": normalize_text(v), "label_name": row["label_name"]})

    df_aug         = pd.DataFrame(aug_rows)
    df_train_final = pd.concat([df_train_sub[["text", "label_name"]], df_aug], ignore_index=True)
    df_train_final = df_train_final.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n Kích thước dữ liệu:")
    print(f"  Train gốc  : {len(df_train_sub):,} mẫu")
    print(f"  Augmented  : +{len(df_aug):,} mẫu ({len(df_aug)/len(df_train_sub)*100:.1f}%)")
    print(f"  Train final: {len(df_train_final):,} mẫu")
    print(f"  Test       : {len(df_test_sub):,} mẫu")

    # 5. Label mapping
    label2id = {l: i for i, l in enumerate(sorted(top_intents))}
    id2label = {i: l for l, i in label2id.items()}

    # 6. Build FEW-SHOT SYSTEM PROMPT
    ex_rows = df_train_sub.groupby("label_name").first().reset_index()
    ex1 = ex_rows.iloc[0]
    ex2 = ex_rows.iloc[1]

    valid_labels_str = "\n".join(f"  {l}" for l in sorted(top_intents))

    SYSTEM_PROMPT = f"""You are a banking intent classifier.
Given a customer message, respond with ONLY the exact intent label.

VALID LABELS (you must use exactly one of these):
{valid_labels_str}

RULES:
- Output ONLY the label name, nothing else
- No explanation, no punctuation, no extra words
- The label must match exactly one from the list above

EXAMPLES:
Customer: "{ex1["text"]}"
Intent: {ex1["label_name"]}

Customer: "{ex2["text"]}"
Intent: {ex2["label_name"]}"""

    print(f"\n SYSTEM_PROMPT ({len(SYSTEM_PROMPT)} chars):")
    print(SYSTEM_PROMPT[:500] + "...")

    # ── Lưu Train/Test & Label Map ───────────────────────────────────
    df_train = df_train_final.rename(columns={"label_name": "label_name"})
    df_test  = df_test_sub.copy()

    df_train.to_csv(f"{CFG['sample_data_dir']}/train.csv", index=False)
    df_test.to_csv(f"{CFG['sample_data_dir']}/test.csv",  index=False)

    label_map = {
        "label2id"     : label2id,
        "id2label"     : {str(k): v for k, v in id2label.items()},
        "system_prompt": SYSTEM_PROMPT,
        "valid_labels" : sorted(top_intents),
    }
    with open(f"{CFG['output_dir']}/label_map.json", "w") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    print(f"\n Đã lưu train.csv ({len(df_train):,} rows), test.csv ({len(df_test):,} rows)")
    print(f" label_map.json với {len(top_intents)} nhãn + system_prompt")
    print(df_train.head(3).to_string())


if __name__ == "__main__":
    main()
