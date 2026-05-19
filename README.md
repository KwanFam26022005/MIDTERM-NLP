# 🎓 MIDTERM NLP — Xử lý Ngôn ngữ Tự nhiên Tiếng Việt

> **Bài thi giữa kỳ môn Xử lý Ngôn ngữ Tự nhiên (NLP)**
>
> **Sinh viên:** Nguyễn Gia Quân (52300102) — Huỳnh Thế Hiệp (52300107)

---

## 📋 Tổng quan

Repository gồm **2 bài toán chính** trong lĩnh vực NLP tiếng Việt:

| Bài | Tên bài toán | Mô hình | Thư mục |
|-----|-------------|---------|---------|
| **Câu 1** | Multi-Task NLP (NER + Sentiment + ABSA) | PhoBERT + CRF | `Cau1/` |
| **Câu 2** | Khôi phục dấu tiếng Việt | LSTM / BiLSTM | `Cau2_new/` |

---

## 📁 Cấu trúc thư mục

```
MIDTERM-NLP/
├── Cau1/                          # Câu 1: Multi-Task NLP
│   ├── main.py                    # Entry point — huấn luyện pipeline
│   ├── model.py                   # MultiTaskPhoBERT (backbone + 3 heads)
│   ├── data_loader.py             # Load & hợp nhất 3 datasets
│   ├── trainer.py                 # Training loop (mixed-precision, multi-task)
│   ├── evaluate.py                # Đánh giá & xuất báo cáo
│   ├── demo_cau1.py               # Gradio demo UI
│   ├── data/                      # Dữ liệu (không push lên git)
│   ├── checkpoints/               # Model checkpoints (không push)
│   └── results/                   # Kết quả đánh giá (không push)
│
├── Cau2_new/                      # Câu 2: Khôi phục dấu tiếng Việt
│   ├── main.py                    # Entry point — pipeline hoàn chỉnh
│   ├── model.py                   # LSTMLanguageModel (Vanilla/Stacked/Bi)
│   ├── preprocessing.py           # Tiền xử lý & tạo nhiễu tiếng Việt
│   ├── vocabulary.py              # Xây dựng & quản lý vocab
│   ├── dataset.py                 # Custom PyTorch Dataset
│   ├── trainer.py                 # Training loop & checkpoint
│   ├── inference.py               # Greedy decoding — khôi phục dấu
│   ├── visualize.py               # Vẽ biểu đồ kết quả
│   ├── demo_cau2.py               # Gradio demo UI
│   └── results/                   # Kết quả & biểu đồ
│
├── Cau2/                          # Câu 2 (phiên bản cũ — tham khảo)
├── .gitignore
└── README.md
```

---

## 🔬 Câu 1: Multi-Task Vietnamese NLP với PhoBERT

### Mô tả bài toán

Xây dựng một mô hình **đa nhiệm (Multi-Task Learning)** dựa trên **PhoBERT** để giải quyết đồng thời 3 bài toán NLP tiếng Việt:

1. **NER (Named Entity Recognition)** — Nhận diện thực thể có tên
2. **Sentiment Analysis** — Phân tích cảm xúc văn bản
3. **ABSA (Aspect-Based Sentiment Analysis)** — Phân tích cảm xúc theo khía cạnh

### Kiến trúc mô hình

```
                    ┌─────────────────────────┐
                    │     PhoBERT Backbone     │
                    │    (vinai/phobert-base)  │
                    └────────┬────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌────────────┐  ┌────────────┐  ┌────────────┐
     │  NER Head  │  │ Sentiment  │  │  ABSA Head │
     │ Linear+CRF │  │ MeanPool+  │  │  [CLS]+    │
     │  (9 tags)  │  │ Linear (3) │  │ Linear(30) │
     └────────────┘  └────────────┘  └────────────┘
```

- **NER Head**: Linear → CRF layer, 9 nhãn IOB2 (O, B/I-PER, B/I-ORG, B/I-LOC, B/I-MISC)
- **Sentiment Head**: Mean Pooling → Dropout → Linear, 3 lớp (Negative, Neutral, Positive)
- **ABSA Head**: [CLS] token → Dropout → Linear, 10 khía cạnh × 3 cực tính

### Datasets

| Dataset | Bài toán | Số nhãn | Nguồn |
|---------|---------|---------|-------|
| **UIT-VSFC** | Sentiment Analysis | 3 (neg/neu/pos) | [GitHub](https://github.com/sonlam1102/UIT-VSFC) |
| **UIT-ViSFD** | ABSA | 10 aspects × 3 polarities | [GitHub](https://github.com/kimkim00/UIT-ViSFD) |
| **VLSP 2016 NER** | Named Entity Recognition | 9 IOB2 tags | [VLSP](https://vlsp.org.vn/resources) |

### Chiến lược huấn luyện

- **Optimizer**: AdamW — backbone lr=2e-5, heads lr=1e-3
- **Scheduler**: Linear warmup 10% → Cosine decay
- **Mixed Precision**: `torch.amp.autocast` (trên GPU)
- **Task Sampling**: NER 30% / Sentiment 40% / ABSA 30%
- **Gradient Clipping**: max_norm = 1.0

### Cách chạy

```bash
cd Cau1

# Huấn luyện
python main.py --data_dir data --epochs 5 --batch_size 16

# Đánh giá
python evaluate.py --ckpt checkpoints/best_ner.pt --data_dir data

# Demo Gradio
python demo_cau1.py
```

---

## 🔤 Câu 2: Khôi phục dấu tiếng Việt bằng LSTM

### Mô tả bài toán

Xây dựng mô hình **Language Model** dựa trên LSTM để **khôi phục dấu** cho văn bản tiếng Việt không dấu (hoặc bị lỗi chính tả, teencode).

**Ví dụ:**
```
Input : "hom nay thoi tiet that dep de di dao pho"
Output: "hôm nay thời tiết thật đẹp để đi dạo phố"
```

### Kiến trúc mô hình

So sánh 3 biến thể LSTM:

| Mô hình | Layers | Bidirectional | Hidden Dim | Embed Dim |
|---------|--------|---------------|------------|-----------|
| **Vanilla LSTM** | 1 | ✗ | 512 | 256 |
| **Stacked LSTM** | 2 | ✗ | 512 | 256 |
| **BiLSTM** | 2 | ✓ | 256 | 256 |

```
  Input (không dấu)
        │
  ┌─────▼─────┐
  │ Embedding  │  (vocab_size × 256)
  │ + Dropout  │
  └─────┬─────┘
        │
  ┌─────▼─────┐
  │   LSTM     │  (Vanilla / Stacked / BiLSTM)
  │ + Dropout  │
  └─────┬─────┘
        │
  ┌─────▼─────┐
  │  Linear    │  (hidden_dim → vocab_size)
  └─────┬─────┘
        │
  Output (có dấu)
```

### Pipeline xử lý dữ liệu

1. **Nguồn dữ liệu**: `comet24082002/vie_wiki_dataset` (HuggingFace)
2. **Tiền xử lý**: Làm sạch HTML, URL, ký tự đặc biệt → lowercase
3. **Tạo nhiễu (Data Augmentation)**:
   - 70% bỏ dấu hoàn toàn (`unidecode`)
   - 10% teencode (`không` → `ko`, `được` → `dc`, `gì` → `j`)
   - 10% sai chính tả ngọng (`tr` ↔ `ch`, `s` ↔ `x`, `l` ↔ `n`)
   - 10% giữ nguyên (để mô hình không quên mặt chữ đúng)
4. **Vocabulary**: Top 12,000 từ phổ biến nhất + 4 special tokens
5. **Phân chia**: Train 90% / Val 5% / Test 5%

### Cách chạy

> ⚠️ **Lưu ý**: Nên chạy trên Google Colab với GPU để tăng tốc huấn luyện.

```bash
cd Cau2_new

# Chạy toàn bộ pipeline (load data → train → evaluate → demo)
python main.py
```

Pipeline sẽ tự động:
1. Tải và tiền xử lý dữ liệu từ HuggingFace
2. Xây dựng vocabulary
3. Huấn luyện 3 mô hình (Vanilla LSTM, Stacked LSTM, BiLSTM)
4. Đánh giá Perplexity trên test set
5. Demo khôi phục dấu
6. Vẽ biểu đồ so sánh

---

## ⚙️ Cài đặt

### Yêu cầu hệ thống

- Python ≥ 3.8
- CUDA GPU (khuyến nghị, không bắt buộc)

### Cài đặt dependencies

```bash
# Câu 1 — Multi-Task PhoBERT
pip install torch transformers datasets seqeval scikit-learn rich pytorch-crf matplotlib gradio tqdm numpy

# Câu 2 — LSTM Diacritics Restoration
pip install torch datasets unidecode matplotlib
```

### Chuẩn bị dữ liệu (Câu 1)

Tải 3 bộ dữ liệu và đặt vào thư mục `Cau1/data/`:

```
Cau1/data/
├── UIT-VSFC/
│   ├── train/
│   │   ├── sents.txt
│   │   └── sentiments.txt
│   ├── dev/
│   └── test/
├── UIT-ViSFD/
│   ├── Train.csv
│   ├── Dev.csv
│   └── Test.csv
└── VLSP-NER/
    ├── train.txt
    ├── dev.txt
    └── test.txt
```

---

## 🖥️ Demo

Cả hai bài đều có giao diện **Gradio** để demo trực quan:

### Câu 1 — Multi-Task Demo

```bash
cd Cau1 && python demo_cau1.py
```

Hỗ trợ 3 tác vụ:
- **NER**: Highlight các thực thể PER, ORG, LOC, MISC
- **Sentiment**: Phân tích cảm xúc với xác suất từng lớp
- **ABSA**: Phân tích cảm xúc theo 10 khía cạnh sản phẩm

### Câu 2 — Khôi phục dấu Demo

```bash
cd Cau2_new
# Demo được tích hợp trong main.py hoặc chạy riêng qua launch_demo()
```

Hỗ trợ chọn mô hình (Vanilla LSTM / Stacked LSTM / BiLSTM) và nhập câu không dấu hoặc teencode.

---

## 🛠️ Công nghệ sử dụng

| Thành phần | Công nghệ |
|-----------|-----------|
| Framework | PyTorch |
| Pretrained Model | PhoBERT (`vinai/phobert-base`) |
| Tokenizer | HuggingFace Transformers |
| NER Decoding | CRF (`pytorch-crf`) |
| Data Processing | HuggingFace Datasets |
| Evaluation | seqeval, scikit-learn |
| Visualization | Matplotlib |
| Demo UI | Gradio |
| Mixed Precision | `torch.amp` |

---

## 📄 License

Dự án phục vụ mục đích học tập — Bài thi giữa kỳ môn NLP.
