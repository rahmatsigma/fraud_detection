import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import os

# --- KONFIGURASI ---
DATA_PATH = 'dataset/creditcard.csv'
MODEL_PATH = 'models/fraud_model.pkl'
SCALER_PATH = 'models/scaler.pkl'

def train_model():
    print("1. Memuat Dataset...")
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: File {DATA_PATH} tidak ditemukan. Download dulu dari Kaggle!")
        return

    df = pd.read_csv(DATA_PATH)
    
    # Cek distribusi kelas
    print(f"   Total Transaksi: {len(df)}")
    print(f"   Jumlah Penipuan (Asli): {len(df[df['Class']==1])}")
    print(f"   Jumlah Aman (Asli): {len(df[df['Class']==0])}")

    # 2. Preprocessing
    print("\n2. Preprocessing Data...")
    
    # Kita perlu menstandarisasi kolom 'Amount' dan 'Time' agar rentang angkanya sama dengan V1-V28
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    
    # Pisahkan Fitur (X) dan Target (y)
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Split data (80% Train, 20% Test)
    # Penting: Split DULU baru SMOTE, agar data validasi tetap murni/asli
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Handling Imbalance dengan SMOTE
    print("\n3. Melakukan SMOTE (Synthetic Minority Over-sampling)...")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    
    print(f"   Data Training sebelum SMOTE: {y_train.value_counts().to_dict()}")
    print(f"   Data Training setelah SMOTE: {y_train_res.value_counts().to_dict()}")
    print("   (Sekarang jumlah penipuan dan aman seimbang di data latihan)")

    # 4. Training Model
    print("\n4. Melatih Random Forest (Bisa memakan waktu 2-5 menit)...")
    # n_estimators=100 artinya menggunakan 100 pohon keputusan
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_res, y_train_res)

    # 5. Evaluasi
    print("\n5. Evaluasi Model pada Data Test (Data Asli)...")
    y_pred = model.predict(X_test)
    
    # Tampilkan Confusion Matrix dan Report
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['Aman', 'Penipuan']))
    
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"   Deteksi Benar Aman: {cm[0][0]}")
    print(f"   Salah Deteksi (False Alarm): {cm[0][1]}")
    print(f"   Gagal Deteksi (Bahaya!): {cm[1][0]}")
    print(f"   Deteksi Benar Penipuan: {cm[1][1]}")

    # 6. Simpan Model
    if not os.path.exists('models'):
        os.makedirs('models')
    
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n✅ Model dan Scaler berhasil disimpan di folder 'models/'")

if __name__ == "__main__":
    train_model()