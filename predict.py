import joblib
import numpy as np
import pandas as pd

# --- KONFIGURASI ---
MODEL_PATH = 'models/fraud_model.pkl'
SCALER_PATH = 'models/scaler.pkl'

def predict_transaction():
    print("Memuat model...")
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError:
        print("❌ Error: Model belum ada. Jalankan train.py dulu!")
        return

    print("--- SIMULASI TRANSAKSI BARU ---")
    
    # Kita akan membuat 2 contoh data dummy
    # Ingat: Input harus ada 30 fitur (Time, V1...V28, Amount)
    
    # Kasus 1: Transaksi Acak (Kemungkinan besar Aman)
    # np.random.randn menghasilkan angka acak distribusi normal
    input_data_aman = np.random.randn(1, 30) 
    
    # Kasus 2: Transaksi Mencurigakan (Kita manipulasi nilai V fitur agar ekstrem)
    input_data_fraud = np.random.randn(1, 30)
    input_data_fraud[0][1] = -50.5  # Contoh nilai V1 yang aneh
    input_data_fraud[0][3] = -20.2  # Contoh nilai V3 yang aneh
    
    # Pilih salah satu untuk dites
    sample_transaction = input_data_aman # Coba ganti jadi input_data_fraud
    
    # Ubah ke DataFrame agar sesuai format training (optional tapi rapi)
    columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    df_input = pd.DataFrame(sample_transaction, columns=columns)

    # Preprocessing (Sama seperti training)
    # Kita harus scale kolom Amount dan Time
    df_input['Amount'] = scaler.transform(df_input['Amount'].values.reshape(-1, 1))
    df_input['Time'] = scaler.transform(df_input['Time'].values.reshape(-1, 1))

    # Prediksi
    prediction = model.predict(df_input)
    probability = model.predict_proba(df_input)

    print(f"\nData Input (Vektor Fitur):")
    print(df_input.iloc[0].values[:5], "... (dan seterusnya)")

    print("\n--- HASIL ANALISIS ---")
    if prediction[0] == 0:
        print(f"✅ Status: TRANSAKSI AMAN")
        print(f"   Probabilitas Aman: {probability[0][0]*100:.2f}%")
    else:
        print(f"🚨 Status: TRANSAKSI MENCURIGAKAN (FRAUD DETECTED!)")
        print(f"   Probabilitas Penipuan: {probability[0][1]*100:.2f}%")

if __name__ == "__main__":
    predict_transaction()