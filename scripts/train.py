import os, json, re, yaml, random, time, glob
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig


def main():
    with open("configs/train.yaml", "r") as f:
        CFG = yaml.safe_load(f)

    os.makedirs(CFG["sample_data_dir"], exist_ok=True)
    os.makedirs(CFG["output_dir"],      exist_ok=True)
    os.makedirs(CFG["configs_dir"],     exist_ok=True)

    # ── Load train/test data ──────────────────────────────────────────────────
    df_train = pd.read_csv(f"{CFG['sample_data_dir']}/train.csv")
    df_test  = pd.read_csv(f"{CFG['sample_data_dir']}/test.csv")
    print(f" Train: {len(df_train):,} | Test: {len(df_test):,}")

    with open(f"{CFG['output_dir']}/label_map.json", "r") as f:
        lmap = json.load(f)
    SYSTEM_PROMPT = lmap["system_prompt"]

    # ── Load Model + LoRA ────────────────────────────────────────────
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = CFG["model_name"],
        max_seq_length = CFG["max_seq_length"],
        load_in_4bit   = CFG["load_in_4bit"],
        dtype          = None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r                          = CFG["lora_r"],
        lora_alpha                 = CFG["lora_alpha"],
        lora_dropout               = CFG["lora_dropout"],
        bias                       = CFG["lora_bias"],
        target_modules             = CFG["target_modules"],
        use_gradient_checkpointing = CFG["use_gradient_checkpointing"],
        random_state               = 42,
        use_rslora                 = False,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"   Model: {CFG['model_name']}")
    print(f"   LoRA r={CFG['lora_r']}, alpha={CFG['lora_alpha']}, dropout={CFG['lora_dropout']}")
    print(f"   Trainable params: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.2f}%)")

    # ── Format Dataset ───────────────────────────────────────────────
    def format_sample(row):
        """Chuyển một sample thành chat format với few-shot system prompt."""
        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": str(row["text"])},
            {"role": "assistant", "content": str(row["label_name"])},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize              = False,
            add_generation_prompt = False,
        )

    df_train["formatted"] = df_train.apply(format_sample, axis=1)

    sample_tokens = tokenizer(df_train["formatted"].iloc[0], return_tensors="pt")
    print(f" Ví dụ sample sau format:")
    print(df_train["formatted"].iloc[0][:500])
    print(f"\n Token count của sample 0: {sample_tokens.input_ids.shape[-1]}")
    print(f"   Max token trong dataset: {df_train['formatted'].map(lambda x: len(tokenizer.encode(x))).max()}")
    print(f"   Mean token: {df_train['formatted'].map(lambda x: len(tokenizer.encode(x))).mean():.0f}")

    hf_train = Dataset.from_dict({"text": df_train["formatted"].tolist()})
    print(f"\n HF Dataset: {len(hf_train)} samples")

    # ── Train ────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model         = model,
        tokenizer     = tokenizer,
        train_dataset = hf_train,
        args = SFTConfig(
            dataset_text_field          = "text",
            per_device_train_batch_size = CFG["per_device_train_batch_size"],
            gradient_accumulation_steps = CFG["gradient_accumulation_steps"],
            num_train_epochs            = CFG["num_epochs"],
            learning_rate               = CFG["learning_rate"],
            optim                       = CFG["optimizer"],
            warmup_ratio                = CFG["warmup_ratio"],
            weight_decay                = CFG["weight_decay"],
            lr_scheduler_type           = CFG["lr_scheduler_type"],
            logging_steps               = CFG["logging_steps"],
            output_dir                  = CFG["output_dir"],
            fp16                        = not torch.cuda.is_bf16_supported(),
            bf16                        = torch.cuda.is_bf16_supported(),
            max_seq_length              = CFG["max_seq_length"],
            packing                     = False,
            save_strategy               = "epoch",
            save_total_limit            = 2,
            report_to                   = "none",
            seed                        = 42,
        ),
    )

    t0 = time.time()
    trainer_stats = trainer.train()
    elapsed = time.time() - t0

    print(f"\n Fine-tuning hoàn tất!")
    print(f"   Thời gian   : {elapsed:.0f}s ({elapsed/60:.1f} phút)")
    print(f"   Final loss  : {trainer_stats.training_loss:.4f}")

    print("\nLoss theo bước:")
    logs = [l for l in trainer.state.log_history if "loss" in l]
    df_loss = pd.DataFrame(logs)[["step", "loss"]].dropna()
    print(df_loss.tail(10).to_string(index=False))

    # ── Lưu Checkpoint + Configs ───────────────────────────────────
    model.save_pretrained(CFG["output_dir"])
    tokenizer.save_pretrained(CFG["output_dir"])

    with open(f"{CFG['configs_dir']}/inference.yaml", "w") as f:
        yaml.dump({
            "model_checkpoint": CFG["model_name"],
            "lora_checkpoint" : CFG["output_dir"],
            "max_seq_length"  : CFG["max_seq_length"],
            "load_in_4bit"    : CFG["load_in_4bit"],
            "label_map_path"  : f"{CFG['output_dir']}/label_map.json",
            "max_new_tokens"  : 20,
        }, f)

    with open(f"{CFG['configs_dir']}/train.yaml", "w") as f:
        yaml.dump({k: v for k, v in CFG.items()}, f)

    print(" LoRA checkpoint:", CFG["output_dir"])
    print(" configs/inference.yaml")
    print(" configs/train.yaml")

    print("\nDanh sách file đã lưu:")
    for fp in sorted(glob.glob(f"{CFG['output_dir']}/**/*", recursive=True)):
        if os.path.isfile(fp) and "checkpoint-" not in fp and "unsloth_compiled" not in fp:
            print(f"  {fp} ({os.path.getsize(fp)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
