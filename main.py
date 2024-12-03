import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# Pemeriksaan dataset yang akan digunakan
try:
    data_jantung = pd.read_csv('dataset/Dataset_Jantung.csv')
except FileNotFoundError:
    st.error("File 'Dataset_Jantung.csv' tidak ditemukan. Periksa path dan lokasi file!")

try:
    data_ginjal = pd.read_csv('dataset/Dataset_Ginjal.csv')
except FileNotFoundError:
    st.error("File 'Dataset_Ginjal.csv' tidak ditemukan. Periksa path dan lokasi file!")


# Menampilkan web
st.title('Klasifikasi Penyakit Jantung, dan Ginjal')
st.write("""
    # Menggunakan beberapa algoritma dan dataset yang berbeda
    #### Mana yang Terbaik?
    """
)

# Menampilkan pilihan dataset
nama_dataset = st.sidebar.selectbox(
    'Pilih Dataset',
    ('Penyakit Jantung', 'Penyakit Ginjal')
)

st.write(f"## Dataset {nama_dataset}")

# menampilkan pilihan algoritma
algoritma = st.sidebar.selectbox(
    'Pilih Algoritma',
    ('KNN', 'SVM', 'Random Forest')
)

# inisialisasi dataset yang akan di tampilkan
def pilih_dataset(nama):
    data = None
    if nama == 'Penyakit Jantung':
        data = data_jantung
        return data
    else:
        data = data_ginjal
        return data

# Preprocessing dataset
def preprocess_dataset(dataset):
    # Periksa nilai kosong
    if dataset.isnull().values.any():
        st.warning("Dataset mengandung nilai kosong. Mengisi nilai kosong dengan rata-rata (untuk numerik) atau mode (untuk kategori).")
        for column in dataset.columns:
            if dataset[column].dtype == 'object':  # Kolom kategori
                dataset[column].fillna(dataset[column].mode()[0], inplace=True)
            else:  # Kolom numerik
                dataset[column].fillna(dataset[column].mean(), inplace=True)
    
    # Identifikasi kolom kategori
    categorical_columns = dataset.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        # Lakukan one-hot encoding atau label encoding
        dataset = pd.get_dummies(dataset, columns=categorical_columns, drop_first=True)
    
    return dataset

# Preprocessing
dataset = pilih_dataset(nama_dataset)

if dataset is None:
    st.error("Dataset tidak ditemukan atau gagal dimuat. Periksa pilihan dataset Anda.")
else:
    # Preprocessing dataset
    dataset = preprocess_dataset(dataset)

    # Pisahkan fitur (x) dan target (y)
    x = dataset.iloc[:, :-1]  # Semua kolom kecuali kolom terakhir
    y = dataset.iloc[:, -1]   # Kolom terakhir

    # Tampilkan informasi dataset
    st.write('Jumlah Baris dan Kolom : ', x.shape)
    st.write('Jumlah Kelas : ', y.nunique())

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

params = tambah_parameter(algoritma)

def pilih_klasifikasi(nama_algoritma, params):
    algorithm = None
    if nama_algoritma == 'KNN':
        algorithm = KNeighborsClassifier(n_neighbors=params['K'])
    elif nama_algoritma == 'SVM':
        algorithm = SVC(C=params['C'])
    else:
        algorithm = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            random_state=12345
        )
    return algorithm

algorithm = pilih_klasifikasi(algoritma, params)

# Menampilkan hasil prediksi
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12345)

algorithm.fit(x_train, y_train)
y_pred = algorithm.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

st.write(f'Algoritma : {algoritma}')
st.write(f'Akurasi : {accuracy}' )

#  PLOT DATASET
# Memproyeksi data kedalam 2 komponen PCA
pca = PCA(2)
x_projected = pca.fit_transform(x)

x1 = x_projected[:, 0]
x2 = x_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')
plt.xlabel('Principal Component 1') # menambahkan label untuk sumbu x
plt.ylabel('Principal Component 2') # menambahkan label untuk sumbu y
plt.colorbar() # menambahkan bar warna

st.pyplot(fig)