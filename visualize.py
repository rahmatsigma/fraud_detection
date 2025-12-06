import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# --- KONFIGURASI ---
MODEL_PATH = 'models/fraud_model.pkl'
DATA_PATH = 'dataset/creditcard.csv'

def visualize_results():
    print("1. Memuat Model dan Data...")
    
    # Cek apakah model ada
    if not os.path.exists(MODEL_PATH):
        print("❌ Error: Model belum ada. Jalankan 'train.py' dulu!")
        return

    # Load Model
    model = joblib.load(MODEL_PATH)
    
    # Kita butuh nama-nama kolom fitur (kecuali 'Class')
    # Cara paling aman adalah baca header csv sebentar
    df_sample = pd.read_csv(DATA_PATH, nrows=1)
    feature_names = df_sample.drop('Class', axis=1).columns.tolist()

    # --- BAGIAN 1: FEATURE IMPORTANCE ---
    print("2. Membuat Grafik Feature Importance...")
    
    # Ambil nilai importance dari Random Forest
    importances = model.feature_importances_
    
    # Buat DataFrame biar rapi
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Urutkan dari yang paling penting ke yang kurang penting
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Plotting
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15), palette='viridis')
    
    plt.title('Top 15 Fitur Paling Penting dalam Mendeteksi Penipuan', fontsize=16)
    plt.xlabel('Tingkat Kepentingan (Importance Score)', fontsize=12)
    plt.ylabel('Nama Fitur (V1-V28, Time, Amount)', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Simpan Gambar
    if not os.path.exists('reports'):
        os.makedirs('reports')
    plt.tight_layout()
    plt.savefig('reports/feature_importance.png')
    print("✅ Grafik berhasil disimpan di 'reports/feature_importance.png'")
    plt.show() # Tampilkan jendela gambar

if __name__ == "__main__":
    visualize_results()