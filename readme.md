
# Klasifikasi Penyakit Jantung, Ginjal, dan Diabetes

Program ini menggunakan algoritma Machine Learning K-Nearest Neighbor (KNN), Support Vector Machine (SVM), dan Random Forest.
Dataset yang di gunakan berasal dari website kaggle sebagai berikut:

Dataset_Jantung : https://www.kaggle.com/datasets/amiramohammedi07/heart-disease-prediction-csv

Dataset_Diabetes : https://www.kaggle.com/datasets/willianoliveiragibin/indians-diabetes-v5

Dataset_Ginjal : https://www.kaggle.com/datasets/ahmad03038/chronic-kidney-disease-diagnosis-dataset




## Tech Stack

**Library:** streamlit, pandas, matplotlib.pyplot, sklearn


## UI

![App Screenshot](https://imgur.com/ndrCQtv.png)

![App Screenshot](https://imgur.com/1UANMix.png)


## Installation

1. Clone the repo
   ```sh
   https://github.com/febrian2108/Klasifikasi-data-penyakit-dengan-algoritma.git
   ```
2. Go to project
   ```sh
   cd Klasifikasi-data-penyakit-dengan-algoritma
   cd app
   ```
3. Build env 
   ```sh
   python -m venv venv
   ```
4. Activate env
   ```sh
   env/scripts/activate
   ```
5. install library
   ```sh
   pip install streamlit
   pip install pandas
   pip install matplotlib
   pip install sklearn
   ```
6. run project
   ```sh
   streamlit run main.py
   ```