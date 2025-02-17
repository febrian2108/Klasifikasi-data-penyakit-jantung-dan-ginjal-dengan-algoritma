# README.md

## Klasifikasi Penyakit Jantung dan Ginjal menggunakan Algoritma Machine Learning BETTER VERSION & STEROIDS VERSION

Aplikasi ini merupakan implementasi klasifikasi penyakit jantung dan ginjal menggunakan beberapa algoritma machine learning populer. Pengguna dapat memilih dataset (penyakit jantung atau ginjal), memilih algoritma klasifikasi (K-Nearest Neighbors, Support Vector Machine, atau Random Forest), serta melihat hasil akurasi dan laporan klasifikasi untuk membandingkan kinerja algoritma-algoritma tersebut.

## Documentation

![App Screenshot](https://imgur.com/fTI38gm.png)

![App Screenshot](https://imgur.com/7nMQr2j.png)

![App Screenshot](https://imgur.com/01HG2dS.png)


## Fitur Aplikasi Original

1. **Dataset Pilihan:**
   - Dataset Penyakit Jantung
   - Dataset Penyakit Ginjal

2. **Algoritma Machine Learning:**
   - **K-Nearest Neighbors (KNN):** Algoritma berbasis kedekatan data dalam ruang fitur.
   - **Support Vector Machine (SVM):** Algoritma klasifikasi berbasis margin yang memisahkan kelas-kelas.
   - **Random Forest:** Algoritma berbasis pohon keputusan yang menggabungkan beberapa model pohon untuk meningkatkan akurasi.

3. **Preprocessing Dataset:**
   - Pengisian nilai kosong dengan rata-rata (untuk kolom numerik) atau mode (untuk kolom kategorikal).
   - One-Hot Encoding untuk kolom kategorikal.

4. **Visualisasi:**
   - Visualisasi hasil projeksi data menggunakan PCA (Principal Component Analysis) ke dalam dua dimensi untuk memudahkan pemahaman distribusi data.

5. **Evaluasi Model:**
   - **Akurasi:** Menghitung akurasi model pada data uji.
   - **Cross-Validation:** Evaluasi model menggunakan teknik cross-validation untuk menghindari overfitting.
   - **Confusion Matrix:** Matriks yang menunjukkan prediksi model terhadap kelas sebenarnya.
   - **Classification Report:** Laporan yang memberikan metrik evaluasi seperti precision, recall, dan f1-score.

---

---
## Fitur aplikasi BETTER VERSION

### 1. ** Tambahkan validasi silang untuk evaluasi yang lebih baik **:
Alih-alih hanya mengandalkan split uji kereta tunggal, menambahkan validasi silang (mis., `Cross_val_score`) akan memberikan perkiraan kinerja model yang lebih baik.

### 2. ** Tuning Hyperparameter **:
Gunakan pencarian grid atau pencarian acak untuk secara otomatis menyetel hyperparameters. Ini akan membantu Anda menemukan hyperparameter terbaik daripada memilihnya secara manual.

### 3. ** Tangani ketidakseimbangan kelas **:
Pertimbangkan menyeimbangkan dataset jika ada ketidakseimbangan kelas, karena beberapa pengklasifikasi dapat berkinerja buruk dalam kasus data yang tidak seimbang (mis., Melalui oversampling, undersampling, atau menggunakan bobot kelas seimbang).

### 4. ** Tambahkan lebih banyak preprocessing **:
- ** Fitur penskalaan **: Untuk algoritma seperti SVM atau KNN, menskalakan fitur (mis., Menggunakan `StandardsCaler`) dapat meningkatkan kinerja.
- ** Pemilihan Fitur **: Anda mungkin ingin menggunakan metode seperti `selectKbest` atau` acakForestClassifier.feature_importances_` untuk memilih fitur penting.

### 5. ** Visualisasikan hasil dan sertakan lebih banyak metrik **:
Selain akurasi, Anda dapat memasukkan metrik evaluasi lain seperti matriks kebingungan, laporan klasifikasi, presisi, penarikan, dan skor F1. Menambahkan kurva ROC atau AUC juga bisa berharga untuk tugas klasifikasi.

### 6. ** Tambahkan lebih banyak opsi data **:
Anda dapat menambahkan opsi bagi pengguna untuk memuat dataset mereka sendiri atau menambahkan set data tambahan seperti diabetes atau kumpulan data kanker untuk membandingkan kinerja pada beberapa set data.

### 7. ** Saran Data Pelatihan Tambahan **:
Untuk kasus ini, Anda dapat menambahkan lebih banyak rekayasa fitur. Misalnya:
- Menambahkan istilah interaksi antara fitur atau fitur polinomial.
- Termasuk data medis lainnya seperti usia, tekanan darah, dll., Dapat membuat model lebih kuat.
---

---
## Fitur Aplikasi STERRROIIIDSS VERSION (BETTER VERSION but in ROIDS)

1. ** Muat data pada permintaan **: Alih -alih memuat seluruh dataset ke dalam memori di muka, kami dapat memuat bagian -bagiannya (mis., Hanya sebagian baris atau kolom) ketika pengguna memintanya. Pendekatan ini meminimalkan penggunaan memori, terutama ketika dataset besar.

2. ** PANDAS `chunksize` untuk pemuatan malas **: Untuk file CSV yang sangat besar, kita dapat memuatnya dalam potongan alih -alih memuat seluruh dataset ke dalam memori. Ini memungkinkan kami untuk membaca bagian -bagian file sesuai kebutuhan.

3. ** Mengoptimalkan Jenis Data **: PANDAS menyediakan opsi untuk mengoptimalkan penggunaan memori dengan mengatur tipe data yang efisien untuk kolom (mis., Mengonversi bilangan bulat menjadi tipe yang lebih kecil seperti `int8`,` int16`, dll., Atau menggunakan `kategori` dtype dtype untuk variabel kategori).

4. ** Penanganan yang efisien dari preprocessing data **: Alih -alih preprocessing seluruh dataset di muka, kita dapat preprocess dengan malas dengan hanya memproses bagian yang relevan dari dataset saat dibutuhkan.
---

## Penjelasan Kode

### 1. **Import Libraries**

```python
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
```

- **Streamlit** digunakan untuk membuat antarmuka web yang interaktif.
- **Pandas** digunakan untuk pengelolaan dataset.
- **Matplotlib** digunakan untuk membuat visualisasi dataset.
- **Scikit-learn (sklearn)** menyediakan berbagai algoritma machine learning dan alat untuk evaluasi dan preprocessing data.

### 2. **Membaca Dataset**

```python
try:
    data_jantung = pd.read_csv('dataset/Dataset_Jantung.csv')
except FileNotFoundError:
    st.error("File 'Dataset_Jantung.csv' tidak ditemukan. Periksa path dan lokasi file!")

try:
    data_ginjal = pd.read_csv('dataset/Dataset_Ginjal.csv')
except FileNotFoundError:
    st.error("File 'Dataset_Ginjal.csv' tidak ditemukan. Periksa path dan lokasi file!")
```

- Aplikasi mencoba untuk memuat dua dataset: **Dataset_Jantung.csv** dan **Dataset_Ginjal.csv**. 
- Jika file tidak ditemukan, aplikasi menampilkan pesan error menggunakan Streamlit.

### 3. **Menampilkan Antarmuka Pengguna**

```python
st.title('Klasifikasi Penyakit Jantung dan Ginjal')
st.write("""
    # Menggunakan beberapa algoritma dan dataset yang berbeda
    #### Mana yang Terbaik?
    """
)
```

- Streamlit digunakan untuk menampilkan judul dan deskripsi aplikasi pada antarmuka web.

### 4. **Memilih Dataset dan Algoritma**

```python
nama_dataset = st.sidebar.selectbox(
    'Pilih Dataset',
    ('Penyakit Jantung', 'Penyakit Ginjal')
)

algoritma = st.sidebar.selectbox(
    'Pilih Algoritma',
    ('KNN', 'SVM', 'Random Forest')
)
```

- Pengguna dapat memilih dataset dan algoritma melalui sidebar di antarmuka.

### 5. **Preprocessing Dataset**

```python
def preprocess_dataset(dataset):
    if dataset.isnull().values.any():
        st.warning("Dataset mengandung nilai kosong. Mengisi nilai kosong dengan rata-rata (untuk numerik) atau mode (untuk kategori).")
        for column in dataset.columns:
            if dataset[column].dtype == 'object':  # Kolom kategori
                dataset[column].fillna(dataset[column].mode()[0], inplace=True)
            else:  # Kolom numerik
                dataset[column].fillna(dataset[column].mean(), inplace=True)

    categorical_columns = dataset.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        dataset = pd.get_dummies(dataset, columns=categorical_columns, drop_first=True)

    return dataset
```

- Fungsi ini memeriksa nilai kosong dalam dataset dan mengisinya dengan rata-rata untuk kolom numerik atau mode untuk kolom kategorikal.
- Kolom kategorikal diubah menjadi format one-hot encoding untuk memudahkan pemodelan.

### 6. **Pemilihan dan Preprocessing Data**

```python
dataset = pilih_dataset(nama_dataset)
dataset = preprocess_dataset(dataset)
```

- Dataset yang dipilih oleh pengguna diproses dengan memanggil fungsi `preprocess_dataset`.

### 7. **Pemisahan Fitur dan Target**

```python
x = dataset.iloc[:, :-1]  # Semua kolom kecuali kolom terakhir
y = dataset.iloc[:, -1]   # Kolom terakhir
```

- Fitur (X) adalah semua kolom kecuali kolom target (y).

### 8. **Feature Scaling**

```python
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
```

- StandardScaler digunakan untuk melakukan normalisasi pada data agar algoritma seperti KNN dan SVM berfungsi lebih baik.

### 9. **Parameter dan Pemilihan Model**

```python
def tambah_parameter(nama_algoritma):
    params = dict()
    if nama_algoritma == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    elif nama_algoritma == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    else:
        max_depth = st.sidebar.slider('max_depth', 1, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params
```

- Fungsi ini menyediakan kontrol untuk pengguna memilih parameter model (seperti jumlah tetangga K untuk KNN, atau parameter C untuk SVM).

### 10. **Evaluasi Model**

```python
cross_val_score_result = cross_val_score(algorithm, x_train, y_train, cv=5)
```

- **Cross-validation** digunakan untuk menilai model dengan membagi data menjadi beberapa lipatan untuk evaluasi yang lebih stabil.

### 11. **Hasil Prediksi dan Evaluasi**

```python
st.write(f'Akurasi : {accuracy}')
st.write(f'Cross-Validation Accuracy: {cross_val_score_result.mean()}')
```

- Akurasi dan hasil evaluasi lainnya ditampilkan di antarmuka Streamlit.

### 12. **Visualisasi Dataset**

```python
pca = PCA(2)
x_projected = pca.fit_transform(x_scaled)

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

st.pyplot(fig)
```

- Hasil dari PCA digunakan untuk mereduksi dimensi dataset dan divisualisasikan dalam bentuk scatter plot untuk dua komponen utama.

---

## Cara Menjalankan Aplikasi

1. Install dependencies
   
2. Jalankan aplikasi dengan:
   ```
   streamlit run app.py
   ```

Pastikan Anda memiliki dataset **Dataset_Jantung.csv** dan **Dataset_Ginjal.csv** di folder `dataset/` agar aplikasi dapat berfungsi dengan baik.

---

## Kesimpulan

Aplikasi ini memberikan gambaran mengenai bagaimana tiga algoritma machine learning yang berbeda dapat digunakan untuk klasifikasi penyakit jantung dan ginjal. Dengan pemrosesan data yang tepat, parameter model yang dapat dikustomisasi, serta evaluasi dan visualisasi, aplikasi ini menyediakan solusi yang mudah digunakan bagi pengguna yang ingin menguji performa algoritma machine learning pada dataset medis.



# Dataset Klasifikasi Penyakit Jantung, dan Ginjal yang digunakan

Dataset_Jantung : https://www.kaggle.com/datasets/amiramohammedi07/heart-disease-prediction-csv

Dataset_Ginjal : https://www.kaggle.com/datasets/ahmad03038/chronic-kidney-disease-diagnosis-dataset

