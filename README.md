#  Analisis Sentimen Terhadap Isu Deforestasi pada Media Sosial Twitter Berbahasa Indonesia 


<img width="944" height="528" alt="image" src="https://github.com/user-attachments/assets/c2fb9aaf-d308-4ec6-91af-a216aa0ae758" />


##  Deskripsi Proyek

Proyek ini mengimplementasikan sentiment analysis pada tweet berbahasa Indonesia dengan topik deforestasi dan pengundulan hutan menggunakan teknik **fine-tuning** pada pre-trained language model IndoBERT. Proyek ini membandingkan performa beberapa model transformer berbasis BERT untuk klasifikasi sentimen teks Indonesia.

### Tujuan
1. Memahami proses fine-tuning pre-trained language model
2. Melakukan hyperparameter tuning untuk optimasi model
3. Mengevaluasi dan membandingkan performa berbagai model IndoBERT
4. Menganalisis sentimen publik terhadap isu deforestasi di Indonesia

##  Dataset

| Atribut | Nilai |
|---------|-------|
| **Sumber** | Twitter (X) |
| **Topik** | Deforestasi & Pengundulan Hutan di Indonesia |
| **Periode** | 20 November - 20 Desember 2025 |
| **Total Tweet** | 1.018tweets |
| **Dataset Final** | 1000 tweets (setelah filter neutral) |
| **Jumlah Kelas** | 2 (Positif, Negatif) |

### Distribusi Kelas
- **Negatif**: 914 tweets (91.4%)
- **Positif**: 86 tweets (8.6%)

> Dataset memiliki ketidakseimbangan kelas yang ditangani dengan **weighted loss function**.

## Preprocessing

Tahapan preprocessing yang dilakukan:

1. **Text Cleaning**
   - Konversi ke lowercase
   - Penghapusan URL
   - Penghapusan mention (@username)
   - Penghapusan hashtag (#)
   - Penghapusan karakter non-alfabet
   - Normalisasi whitespace

2. **Filtering**
   - Menghapus tweet dengan kurang dari 5 kata

3. **Tokenisasi**
   - Menggunakan tokenizer dari masing-masing pre-trained model
   - Max sequence length: 128 tokens
   - Padding: max_length
   - Truncation: True

4. **Data Split**
   - Training: 70%
   - Validation: 15%
   - Testing: 15%

## Model yang Digunakan
Model untuk Hyperparameter Tuning

|No| Model |  Model ID (Hugging Face) |
|----|-------|-------------------------|
| 1 | indoBERT | `indobenchmark/indobert-base-p1` |

Model RoBERTa Indonesian digunakan untuk mencari kombinasi hyperparameter terbaik (12 eksperimen).

### Model untuk Perbandingan

| No | Model | Model ID (Hugging Face) |
|----|-------|-------------------------|
| 1 | IndoBERT (IndoLEM) | `indolem/indobert-base-uncased` |
| 2 | IndoBERT (IndoBenchmark) | `indobenchmark/indobert-base-p1` |
| 3 | mBERT Multilingual | `bert-base-multilingual-uncased` |

### Pendekatan Fine-tuning
- **Metode**: Full Fine-tuning
- **Task**: Sequence Classification (Binary)
- **Framework**: Hugging Face Transformers

## Hyperparameter Tuning

### Parameter yang Diuji

| Parameter | Nilai yang Diuji |
|-----------|------------------|
| Learning Rate | 1e-5, 2e-5, 5e-5 |
| Batch Size | 8, 16 |
| Epochs | 3, 5 |
| Weight Decay | 0.01 |
| Max Length | 128 |

**Total Kombinasi**: 3 × 2 × 2 = **12 eksperimen**

### Justifikasi Hyperparameter

| Parameter | Nilai Optimal | Justifikasi |
|-----------|---------------|-------------|
| **Learning Rate** | 5e-5 | Learning rate yang lebih tinggi (5e-5) memberikan konvergensi lebih cepat pada dataset kecil tanpa menyebabkan overfitting |
| **Batch Size** | 8 | Batch size kecil memberikan gradient yang lebih noisy namun membantu generalisasi pada dataset terbatas |
| **Epochs** | 5 | Cukup untuk konvergensi tanpa overfitting, ditunjukkan oleh validation loss yang stabil |

### Konfigurasi Terbaik
```
Learning Rate: 5e-05
Batch Size: 8
Epochs: 5
Validation Accuracy: 88.89%
```

## Hasil Eksperimen

### Perbandingan Model (dengan konfigurasi optimal)

| Model | Val Accuracy | Val Loss | Train Loss |
|-------|--------------|----------|------------|
| **IndoBenchmark-IndoBERT** | **93.33%** | 1.059707 | 0.529429 |
| mBERT-Multilingual | 92.66% | 1.739180 | 0.807487 |
| IndoLEM-IndoBERT | 91.33% | 0.770590 | 0.746658 |

### Hasil Hyperparameter Tuning (RoBERTa Indonesian)

| Exp | LR | Batch | Epochs | Val Acc | Val Loss |
|-----|-----|-------|--------|---------|----------|
| 10 | 5e-05 | 8 | 5 | **92.67%** | 1.1340 |
| 2 | 1e-05 | 8 | 5 | 92% | 1.4792 |
| 12 | 5e-05 | 16 | 5 | 92% | 2.3006 |
| 9 | 5e-05 | 8 | 3 | 92% | 0.9178 |
| 11 | 5e-05 | 16 | 3 | 92% | 1.5456 |

### Visualisasi

Proyek ini menghasilkan visualisasi:
- Word Cloud dari dataset
- Distribusi sentimen
- Confusion Matrix
- Grafik perbandingan model

## Cara Penggunaan

### Prerequisites

```bash
# Clone repository
git clone https://github.com/username/text-analytics-deforestasi.git
cd text-analytics-deforestasi

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0.0
transformers>=4.40.0
datasets>=2.19.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
wordcloud>=1.9.0
```

### Menjalankan Notebook

1. Upload notebook ke Google Colab
2. Pastikan runtime menggunakan **GPU** (T4 atau lebih tinggi)
3. Jalankan seluruh cell secara berurutan


## Analisis Hasil

### Temuan Utama

1. **IndoBenchmark-IndoBERT**menunjukkan performa terbaik dengan akurasi validasi mencapai 93,33%, mengungguli model IndoLEM-IndoBERT (92,67%) dan model multilingual mBERT (91,33%).

2. **Learning rate 5e-5** merupakan nilai optimal dari rentang yang diuji (1e-5 hingga 5e-5), memberikan akurasi validasi tertinggi untuk proses fine-tuning.

3. **Batch size 8** lebih efektif dibanding 16 pada dataset terbatas, karena memberikan update gradient yang lebih frequent.

4. **5 epochs** diperlukan untuk mencapai akurasi optimal pada konfigurasi terbaik, menunjukkan bahwa model membutuhkan durasi tersebut untuk konvergensi tanpa mengalami overfitting yang signifikan.

### Limitasi

- Dataset relatif kecil dengan hanya 1.000 tweet yang memenuhi kriteria filter minimal 5 kata.
- Ketidakseimbangan kelas yang ekstrem, di mana sentimen negatif mendominasi sebesar 91,4% dibandingkan sentimen positif yang hanya 8,6%.
- Pelabelan otomatis menggunakan metode Context-Aware Pattern Matching (regex), bukan melalui anotasi manual oleh pakar, yang berisiko pada akurasi label dasar.

### Rekomendasi

1. Memperluas Dataset: Mengumpulkan volume data yang lebih besar untuk meningkatkan kemampuan generalisasi model terhadap berbagai variasi opini publik.
2. Validasi Manual: Melakukan anotasi manual (manual labeling) pada sebagian dataset untuk memastikan akurasi label dan memvalidasi efektivitas sistem pattern matching.
3. Augmentasi Data: Mencoba teknik augmentasi data khusus bahasa Indonesia untuk menambah jumlah sampel pada kelas minoritas (Positif) guna menyeimbangkan distribusi kelas.
4. Eksplorasi Model Kontemporer: Mempertimbangkan penggunaan varian model yang lebih besar atau arsitektur lain yang dioptimalkan untuk bahasa Indonesia guna menangkap konteks slang dan singkatan Twitter secara lebih mendalam.

##  Referensi

1. [Fine-Tuning BERT for Text Classification](https://huggingface.co/docs/transformers/training)
2. [IndoBERT: Pre-trained Language Model for Indonesian Language](https://github.com/indobenchmark/indonlu)
3. [IndoLEM: A Benchmark for Indonesian NLP](https://indolem.github.io/)
4. [Hugging Face Model Hub](https://huggingface.co/models)
5. https://www.youtube.com/watch?v=n5Y0RJ5OkdE
