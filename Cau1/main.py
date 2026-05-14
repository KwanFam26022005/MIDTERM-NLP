import os
import argparse
from pathlib import Path

# Cần cài đặt trước: pip install transformers datasets seqeval scikit-learn rich pytorch-crf
from transformers import AutoTokenizer

# Các module từ dự án
from data_loader import load_all_datasets, build_task_dataloader, get_class_weights
from model import MultiTaskPhoBERT
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="MultiTaskPhoBERT Main Training Pipeline")
    parser.add_argument("--data_dir", type=str, default="data", help="Thư mục chứa dữ liệu")
    parser.add_argument("--epochs", type=int, default=5, help="Số lượng epochs huấn luyện")
    parser.add_argument("--batch_size", type=int, default=16, help="Kích thước batch size")
    args = parser.parse_args()

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
        print(f"Vui lòng đảm bảo bạn đã copy thư mục data vào đúng vị trí (ví dụ: !cp -r /content/drive/MyDrive/midterm_nlp/Cau1/data /content/MIDTERM-NLP/Cau1/)")
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
