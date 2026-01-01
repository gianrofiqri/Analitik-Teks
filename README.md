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
| **Total Tweet** | 1.016 tweets |
| **Setelah Preprocessing** | 343 tweets |
| **Dataset Final** | 240 tweets (setelah filter neutral) |
| **Jumlah Kelas** | 2 (Positif, Negatif) |

### Distribusi Kelas
- **Negatif**: 185 tweets (77.1%)
- **Positif**: 55 tweets (22.9%)

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
| 1 | RoBERTa Indonesian | `cahya/roberta-base-indonesian-1.5G` |

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
| **Epochs** | 3 | Cukup untuk konvergensi tanpa overfitting, ditunjukkan oleh validation loss yang stabil |

### Konfigurasi Terbaik
```
Learning Rate: 5e-05
Batch Size: 8
Epochs: 3
Validation Accuracy: 88.89%
```

## Hasil Eksperimen

### Perbandingan Model (dengan konfigurasi optimal)

| Model | Val Accuracy | Val Loss | Train Loss |
|-------|--------------|----------|------------|
| **IndoBenchmark-IndoBERT** | **86.11%** | 0.7940 | 0.3166 |
| mBERT-Multilingual | 80.56% | 0.6772 | 0.6999 |
| IndoLEM-IndoBERT | 77.78% | 0.7901 | 0.6650 |

### Hasil Hyperparameter Tuning (RoBERTa Indonesian)

| Exp | LR | Batch | Epochs | Val Acc | Val Loss |
|-----|-----|-------|--------|---------|----------|
| 7 | 5e-05 | 8 | 3 | **88.89%** | 0.4173 |
| 8 | 5e-05 | 8 | 5 | 86.11% | 0.5232 |
| 2 | 1e-05 | 8 | 5 | 86.11% | 0.5355 |
| 9 | 5e-05 | 16 | 3 | 83.33% | 0.4831 |
| 10 | 5e-05 | 16 | 5 | 83.33% | 0.5567 |

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

1. **IndoBenchmark-IndoBERT** menunjukkan performa terbaik dengan akurasi validasi **86.11%**, mengungguli model multilingual (mBERT) dan IndoLEM-IndoBERT.

2. **Learning rate 5e-5** optimal untuk dataset kecil, memberikan konvergensi cepat tanpa overfitting.

3. **Batch size 8** lebih efektif dibanding 16 pada dataset terbatas, karena memberikan update gradient yang lebih frequent.

4. **3 epochs** sudah cukup untuk mencapai konvergensi optimal, training lebih lanjut cenderung menyebabkan overfitting.

### Limitasi

- Dataset relatif kecil (240 samples setelah filtering)
- Ketidakseimbangan kelas yang signifikan (77% Negatif vs 23% Positif)
- Pelabelan sentimen menggunakan rule-based approach, bukan manual annotation

### Rekomendasi

1. Menggunakan dataset yang lebih besar untuk meningkatkan generalisasi
2. Melakukan manual annotation untuk label yang lebih akurat
3. Mencoba teknik data augmentation untuk mengatasi class imbalance
4. Eksplorasi model IndoBERT-Large atau XLM-RoBERTa untuk performa lebih baik

##  Referensi

1. [Fine-Tuning BERT for Text Classification](https://huggingface.co/docs/transformers/training)
2. [IndoBERT: Pre-trained Language Model for Indonesian Language](https://github.com/indobenchmark/indonlu)
3. [IndoLEM: A Benchmark for Indonesian NLP](https://indolem.github.io/)
4. [Hugging Face Model Hub](https://huggingface.co/models)
5. https://www.youtube.com/watch?v=n5Y0RJ5OkdE
