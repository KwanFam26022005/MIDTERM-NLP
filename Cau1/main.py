import os
import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

# Cần cài đặt trước: pip install transformers datasets seqeval scikit-learn rich pytorch-crf
from transformers import AutoTokenizer

# Các module từ dự án
from data_loader import load_all_datasets, build_task_dataloader, get_class_weights
from model import MultiTaskPhoBERT
from trainer import Trainer


def _dir_ready(path: str) -> bool:
    """Kiểm tra thư mục đã có dữ liệu chưa."""
    return os.path.isdir(path) and len(os.listdir(path)) > 0


def download_data(data_dir: str):
    """
    Tự động tải dữ liệu từ HuggingFace Hub và GitHub ZIP.
    Không sử dụng git clone (tránh lỗi auth trên Colab).

    Nguồn dữ liệu:
      1. UIT-VSFC (Sentiment)  → HuggingFace: uitnlp/vietnamese_students_feedback
      2. UIT-ViSFD (ABSA)      → GitHub ZIP:  LuongPhan/UIT-ViSFD
      3. WikiANN-vi (NER)      → HuggingFace: wikiann (lang=vi)
    """
    from datasets import load_dataset

    os.makedirs(data_dir, exist_ok=True)

    # ─── 1. UIT-VSFC (Sentiment) ─────────────────────────────────────────
    vsfc_dir = os.path.join(data_dir, "UIT-VSFC")
    if not _dir_ready(vsfc_dir):
        print("[Download] Đang tải UIT-VSFC từ HuggingFace Hub...")
        try:
            ds = load_dataset("uitnlp/vietnamese_students_feedback", trust_remote_code=True)
            os.makedirs(vsfc_dir, exist_ok=True)

            split_map = {"train": "train.txt", "validation": "dev.txt", "test": "test.txt"}
            for split_name, fname in split_map.items():
                if split_name not in ds:
                    continue
                fpath = os.path.join(vsfc_dir, fname)
                count = 0
                with open(fpath, "w", encoding="utf-8") as f:
                    for row in ds[split_name]:
                        text = row.get("sentence", row.get("text", "")).strip()
                        label = row.get("sentiment", row.get("label", 0))
                        if text:
                            f.write(f"{text}\t{label}\n")
                            count += 1
                print(f"  ✅ {fname}: {count} mẫu")
        except Exception as e:
            print(f"  ❌ Lỗi tải UIT-VSFC: {e}")
    else:
        print("[Download] UIT-VSFC đã tồn tại.")

    # ─── 2. UIT-ViSFD (ABSA) — tải ZIP từ GitHub ─────────────────────────
    visfd_dir = os.path.join(data_dir, "UIT-ViSFD")
    if not _dir_ready(visfd_dir):
        print("[Download] Đang tải UIT-ViSFD từ GitHub (ZIP)...")
        zip_url = "https://github.com/LuongPhan/UIT-ViSFD/archive/refs/heads/main.zip"
        zip_path = os.path.join(data_dir, "_visfd_tmp.zip")
        tmp_extract = os.path.join(data_dir, "_visfd_tmp")
        try:
            urllib.request.urlretrieve(zip_url, zip_path)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(tmp_extract)
            os.makedirs(visfd_dir, exist_ok=True)

            # Tìm file JSON trong thư mục giải nén
            for root, _, files in os.walk(tmp_extract):
                for file in files:
                    low = file.lower()
                    if low.endswith(".json") and low in [
                        "train.json", "dev.json", "test.json", "val.json",
                    ]:
                        shutil.copy(os.path.join(root, file),
                                    os.path.join(visfd_dir, low))

            # val.json → dev.json nếu cần
            val_j = os.path.join(visfd_dir, "val.json")
            dev_j = os.path.join(visfd_dir, "dev.json")
            if os.path.exists(val_j) and not os.path.exists(dev_j):
                os.rename(val_j, dev_j)

            found = [f for f in os.listdir(visfd_dir) if f.endswith(".json")]
            if found:
                print(f"  ✅ UIT-ViSFD: {found}")
            else:
                print("  ⚠️ Không tìm thấy file JSON trong repo ZIP.")
        except Exception as e:
            print(f"  ❌ Lỗi tải UIT-ViSFD: {e}")
        finally:
            if os.path.exists(zip_path):
                os.remove(zip_path)
            shutil.rmtree(tmp_extract, ignore_errors=True)
    else:
        print("[Download] UIT-ViSFD đã tồn tại.")

    # ─── 3. WikiANN-vi (NER) — thay thế VLSP-NER ─────────────────────────
    vlsp_dir = os.path.join(data_dir, "VLSP-NER")
    if not _dir_ready(vlsp_dir):
        print("[Download] Đang tải WikiANN-vi (NER) từ HuggingFace Hub...")
        try:
            ds = load_dataset("wikiann", "vi", trust_remote_code=True)
            os.makedirs(vlsp_dir, exist_ok=True)

            # WikiANN tag mapping: 0=O, 1=B-PER, 2=I-PER, 3=B-ORG, 4=I-ORG, 5=B-LOC, 6=I-LOC
            WIKIANN_TAGS = [
                "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC",
            ]

            split_map = {"train": "train.txt", "validation": "dev.txt", "test": "test.txt"}
            for split_name, fname in split_map.items():
                if split_name not in ds:
                    continue
                fpath = os.path.join(vlsp_dir, fname)
                count = 0
                with open(fpath, "w", encoding="utf-8") as f:
                    for row in ds[split_name]:
                        tokens = row["tokens"]
                        ner_tags = row["ner_tags"]
                        for tok, tag_id in zip(tokens, ner_tags):
                            tag = WIKIANN_TAGS[tag_id] if tag_id < len(WIKIANN_TAGS) else "O"
                            f.write(f"{tok} {tag}\n")
                        f.write("\n")
                        count += 1
                print(f"  ✅ {fname}: {count} câu")
        except Exception as e:
            print(f"  ❌ Lỗi tải WikiANN-vi: {e}")
    else:
        print("[Download] VLSP-NER (WikiANN-vi) đã tồn tại.")

    print("[Download] ✅ Hoàn tất quá trình chuẩn bị dữ liệu!")

def main():
    parser = argparse.ArgumentParser(description="MultiTaskPhoBERT Main Training Pipeline")
    parser.add_argument("--data_dir", type=str, default="data", help="Thư mục chứa dữ liệu")
    parser.add_argument("--epochs", type=int, default=5, help="Số lượng epochs huấn luyện")
    parser.add_argument("--batch_size", type=int, default=16, help="Kích thước batch size")
    parser.add_argument("--download", action="store_true", help="Bật chế độ tự động tải data")
    args = parser.parse_args()

    # 1. Tự động tải data
    if args.download:
        download_data(args.data_dir)

    print("\n--- KHỞI TẠO TOKENIZER & TẢI DATASETS ---")
    model_name = "vinai/phobert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    try:
        dataset = load_all_datasets(
            vsfc_dir=os.path.join(args.data_dir, "UIT-VSFC"),
            visfd_dir=os.path.join(args.data_dir, "UIT-ViSFD"),
            vlsp_ner_dir=os.path.join(args.data_dir, "VLSP-NER"),
        )
    except FileNotFoundError as e:
        print(f"\n❌ LỖI TẢI DỮ LIỆU: {e}")
        print("Vui lòng chạy lại với tham số --download để tự động tải dữ liệu (ví dụ: python main.py --download)")
        return

    print("\n--- CHUẨN BỊ DATALOADERS ---")
    train_loaders = {
        "ner": build_task_dataloader(dataset["train"], "ner", tokenizer, batch_size=args.batch_size),
        "sentiment": build_task_dataloader(dataset["train"], "sentiment", tokenizer, batch_size=args.batch_size),
        "absa": build_task_dataloader(dataset["train"], "absa", tokenizer, batch_size=args.batch_size),
    }
    val_loaders = {
        "ner": build_task_dataloader(dataset["val"], "ner", tokenizer, batch_size=args.batch_size, shuffle=False),
        "sentiment": build_task_dataloader(dataset["val"], "sentiment", tokenizer, batch_size=args.batch_size, shuffle=False),
        "absa": build_task_dataloader(dataset["val"], "absa", tokenizer, batch_size=args.batch_size, shuffle=False),
    }

    print("\n--- TÍNH TOÁN CLASS WEIGHTS ---")
    class_weights = {
        "sentiment": get_class_weights(dataset["train"], "sentiment"),
        "absa": get_class_weights(dataset["train"], "absa"),
        "ner": get_class_weights(dataset["train"], "ner"),
    }

    print("\n--- KHỞI TẠO MÔ HÌNH VÀ TRAINER ---")
    model = MultiTaskPhoBERT(model_name=model_name)
    
    trainer = Trainer(
        model=model,
        train_loaders=train_loaders,
        val_loaders=val_loaders,
        class_weights=class_weights,
        backbone_lr=2e-5,
        head_lr=1e-3,
        max_grad_norm=1.0,
        warmup_ratio=0.1
    )

    print("\n--- BẮT ĐẦU HUẤN LUYỆN ---")
    trainer.train(epochs=args.epochs, save_dir='checkpoints')

if __name__ == "__main__":
    main()
