import os
import argparse
import subprocess
import shutil
from pathlib import Path

# Cần cài đặt trước: pip install transformers datasets seqeval scikit-learn rich pytorch-crf
from transformers import AutoTokenizer

# Các module từ dự án
from data_loader import load_all_datasets, build_task_dataloader, get_class_weights
from model import MultiTaskPhoBERT
from trainer import Trainer

def download_data(data_dir: str):
    """
    Tự động tải dữ liệu từ các repository GitHub và sắp xếp vào đúng thư mục.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. UIT-VSFC
    vsfc_dir = os.path.join(data_dir, "UIT-VSFC")
    if not os.path.exists(vsfc_dir) or not os.listdir(vsfc_dir):
        print("[Download] Đang tải dataset UIT-VSFC...")
        subprocess.run(["git", "clone", "https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback", "tmp_vsfc"], check=False)
        os.makedirs(vsfc_dir, exist_ok=True)
        # Tìm các file txt (train, dev, test)
        for root, _, files in os.walk("tmp_vsfc"):
            for file in files:
                if file.endswith(".txt") and file.lower() in ["train.txt", "dev.txt", "test.txt", "val.txt"]:
                    shutil.copy(os.path.join(root, file), os.path.join(vsfc_dir, file.lower()))
        
        # Đổi tên val.txt thành dev.txt nếu cần
        if os.path.exists(os.path.join(vsfc_dir, "val.txt")) and not os.path.exists(os.path.join(vsfc_dir, "dev.txt")):
            os.rename(os.path.join(vsfc_dir, "val.txt"), os.path.join(vsfc_dir, "dev.txt"))
        
        shutil.rmtree("tmp_vsfc", ignore_errors=True)
    else:
        print("[Download] UIT-VSFC đã tồn tại.")

    # 2. UIT-ViSFD
    visfd_dir = os.path.join(data_dir, "UIT-ViSFD")
    if not os.path.exists(visfd_dir) or not os.listdir(visfd_dir):
        print("[Download] Đang tải dataset UIT-ViSFD...")
        subprocess.run(["git", "clone", "https://github.com/LuongPhan/UIT-ViSFD.git", "tmp_visfd"], check=False)
        os.makedirs(visfd_dir, exist_ok=True)
        # Tìm các file json (train, dev, test)
        for root, _, files in os.walk("tmp_visfd"):
            for file in files:
                if file.endswith(".json") and file.lower() in ["train.json", "dev.json", "test.json", "val.json"]:
                    shutil.copy(os.path.join(root, file), os.path.join(visfd_dir, file.lower()))
                    
        # Đổi tên val.json thành dev.json nếu cần
        if os.path.exists(os.path.join(visfd_dir, "val.json")) and not os.path.exists(os.path.join(visfd_dir, "dev.json")):
            os.rename(os.path.join(visfd_dir, "val.json"), os.path.join(visfd_dir, "dev.json"))
            
        shutil.rmtree("tmp_visfd", ignore_errors=True)
    else:
        print("[Download] UIT-ViSFD đã tồn tại.")

    # 3. VLSP-NER 2016
    vlsp_dir = os.path.join(data_dir, "VLSP-NER")
    if not os.path.exists(vlsp_dir) or not os.listdir(vlsp_dir):
        print("[Download] Đang tải dataset VLSP-NER (bản copy public)...")
        # Sử dụng một bản copy public phổ biến vì VLSP yêu cầu đăng ký
        subprocess.run(["git", "clone", "https://github.com/NganDong/VLSP2016-NER.git", "tmp_vlsp"], check=False)
        os.makedirs(vlsp_dir, exist_ok=True)
        
        for root, _, files in os.walk("tmp_vlsp"):
            for file in files:
                if file.lower() in ["train.txt", "dev.txt", "test.txt", "val.txt"]:
                    shutil.copy(os.path.join(root, file), os.path.join(vlsp_dir, file.lower()))
        
        if os.path.exists(os.path.join(vlsp_dir, "val.txt")) and not os.path.exists(os.path.join(vlsp_dir, "dev.txt")):
            os.rename(os.path.join(vlsp_dir, "val.txt"), os.path.join(vlsp_dir, "dev.txt"))
            
        shutil.rmtree("tmp_vlsp", ignore_errors=True)
        
        # Cảnh báo nếu không tìm thấy dữ liệu (do repo trên có thể bị lỗi)
        if not os.listdir(vlsp_dir):
            print("⚠️ CẢNH BÁO: Không thể tự động tải VLSP-NER. Bạn hãy tải thủ công từ https://vlsp.org.vn/resources và đặt vào data/VLSP-NER/")
    else:
        print("[Download] VLSP-NER đã tồn tại.")
        
    print("[Download] Hoàn tất quá trình chuẩn bị dữ liệu!")

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
