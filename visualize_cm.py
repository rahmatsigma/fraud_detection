import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# --- KONFIGURASI ---
DATA_PATH = 'dataset/creditcard.csv'
MODEL_PATH = 'models/fraud_model.pkl'
SCALER_PATH = 'models/scaler.pkl'

def plot_confusion_matrix():
    print("1. Mempersiapkan Data Test...")
    
    # Load Data
    if not os.path.exists(DATA_PATH):
        print("❌ Error: Dataset tidak ditemukan.")
        return
    
    df = pd.read_csv(DATA_PATH)

    # Load Scaler yang sudah dilatih (PENTING: Gunakan scaler yang sama dengan training)
    scaler = joblib.load(SCALER_PATH)
    
    # Preprocessing (Sama seperti training)
    df['Amount'] = scaler.transform(df['Amount'].values.reshape(-1, 1))
    df['Time'] = scaler.transform(df['Time'].values.reshape(-1, 1))
    
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Split Data
    # PENTING: random_state harus sama dengan train.py (42) 
    # supaya data test yang dipakai di sini sama persis dengan yang dipakai saat evaluasi
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("2. Melakukan Prediksi...")
    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)

    print("3. Membuat Plot Confusion Matrix...")
    # Hitung Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot menggunakan Seaborn Heatmap
    plt.figure(figsize=(8, 6))
    
    # annot=True (tampilkan angka), fmt='d' (format integer/bilangan bulat), cmap='Blues' (warna biru)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Prediksi Aman', 'Prediksi Penipuan'],
                yticklabels=['Asli Aman', 'Asli Penipuan'])

    plt.title('Confusion Matrix: Deteksi Penipuan Kartu Kredit', fontsize=16)
    plt.ylabel('Label Sebenarnya (Actual)', fontsize=12)
    plt.xlabel('Label Prediksi (Predicted)', fontsize=12)

    # Simpan Gambar
    if not os.path.exists('reports'):
        os.makedirs('reports')
        
    save_path = 'reports/confusion_matrix.png'
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Gambar berhasil disimpan di '{save_path}'")
    
    plt.show()

if __name__ == "__main__":
    plot_confusion_matrix()