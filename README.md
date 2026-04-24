# Banking Intent Detection with Unsloth (QLoRA)

Fine-tune Qwen2.5-7B-Instruct với QLoRA 4-bit để phân loại 20 intent trong lĩnh vực ngân hàng sử dụng dataset BANKING77.

## Kết quả

| Metric | Value |
|---|---|
| Model | Qwen2.5-7B-Instruct (QLoRA 4-bit) |
| Phương pháp | LoRA (r=32, alpha=64) + SFTTrainer |
| Số classes | 20 intents |
| Test Accuracy | **95.00%** |
| Thời gian train | ~574 phút (Kaggle T4 x2) |

## Cấu trúc thư mục

```
banking-intent-unsloth/
├── scripts/
│   ├── preprocess_data.py   # Tiền xử lý & augmentation
│   ├── train.py             # Fine-tuning với Unsloth + SFTTrainer
│   └── inference.py         # Class IntentClassification + evaluation
├── configs/
│   ├── train.yaml           # Hyperparameters
│   └── inference.yaml       # Config cho inference
├── sample_data/
│   ├── train.csv            # 7,585 mẫu (gốc + augmented)
│   └── test.csv             # 800 mẫu
├── train.sh
├── inference.sh
├── requirements.txt
└── README.md
```

## Cài đặt

```bash
pip install -r requirements.txt
```

> **Lưu ý:** Cần chạy trên môi trường Kaggle, nếu chạy trên môi trường khác, phải thay đổi nội dung trong file requirement.txt từ kaggle-new thành môi trường phù hợp.

## Tiền xử lý dữ liệu

```bash
python scripts/preprocess_data.py
```

Script này sẽ:
- Tải dataset BANKING77 từ HuggingFace
- Chọn 20 intent phổ biến nhất
- Normalize text + Data augmentation (~35%)
- Lưu `sample_data/train.csv`, `sample_data/test.csv`, `banking-intent-lora/label_map.json`

## Training

```bash
bash train.sh
# hoặc
python scripts/train.py
```

Checkpoint sẽ được lưu tại `./banking-intent-lora/`.

## Inference

```bash
bash inference.sh
# hoặc
python scripts/inference.py configs/inference.yaml sample_data/test.csv
```

## Video Demo kết quả Inference

[Link Google Drive](https://drive.google.com/file/d/1nywHCVcIiSUtQbs06yLV5BviwqP1syCD/view?usp=sharing)



