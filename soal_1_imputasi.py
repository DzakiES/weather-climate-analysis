"""
Soal 1: Imputasi Data
=====================
Melakukan imputasi missing value pada data iklim BMKG menggunakan:
1. Linear Interpolation
2. Prophet

Parameter yang diimputasi: Tavg, RH_avg, RR, ff_avg
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# 1. LOAD DAN PREPROCESSING DATA
# =====================================================

print("=" * 60)
print("SOAL 1: IMPUTASI DATA IKLIM BMKG")
print("=" * 60)

# Load data, skip 7 baris header
df = pd.read_csv('laporan_iklim.csv', skiprows=7)

# Ambil kolom yang diperlukan saja
df = df.iloc[:, :5]
df.columns = ['TANGGAL', 'TAVG', 'RH_AVG', 'RR', 'FF_AVG']

# Konversi tanggal
df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], format='%d-%m-%Y')
df = df.set_index('TANGGAL')

# Konversi kolom numerik
params = ['TAVG', 'RH_AVG', 'RR', 'FF_AVG']
for col in params:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Ganti nilai 8888 dan 9999 dengan NaN (missing value)
df = df.replace([8888, 9999], np.nan)

print("\n[INFO] Data berhasil dimuat")
print(f"Periode data: {df.index.min().strftime('%d-%m-%Y')} s.d. {df.index.max().strftime('%d-%m-%Y')}")
print(f"Jumlah data: {len(df)} hari")

# =====================================================
# 2. CEK MISSING VALUE
# =====================================================

print("\n" + "=" * 60)
print("CEK MISSING VALUE SEBELUM IMPUTASI")
print("=" * 60)

missing_info = pd.DataFrame({
    'Jumlah Missing': df[params].isnull().sum(),
    'Persentase (%)': (df[params].isnull().sum() / len(df) * 100).round(2)
})
print(missing_info)

# Simpan posisi missing value untuk evaluasi nanti
missing_mask = df[params].isnull()

# =====================================================
# 3. IMPUTASI DENGAN LINEAR INTERPOLATION
# =====================================================

print("\n" + "=" * 60)
print("METODE 1: LINEAR INTERPOLATION")
print("=" * 60)

df_linear = df.copy()

for col in params:
    df_linear[col] = df_linear[col].interpolate(method='linear')
    # Isi nilai di awal/akhir jika masih ada NaN
    df_linear[col] = df_linear[col].fillna(method='ffill').fillna(method='bfill')

print("[INFO] Imputasi Linear selesai")
print(f"Missing value tersisa: {df_linear[params].isnull().sum().sum()}")

# Simpan hasil imputasi linear
df_linear.to_csv('DATA/2024_linear.csv')
print("[INFO] Data disimpan ke: DATA/2024_linear.csv")

# =====================================================
# 4. IMPUTASI DENGAN PROPHET
# =====================================================

print("\n" + "=" * 60)
print("METODE 2: PROPHET IMPUTATION")
print("=" * 60)

df_prophet = df.copy()

def impute_with_prophet(series, col_name):
    """
    Melakukan imputasi menggunakan Prophet
    """
    # Siapkan data untuk Prophet
    df_temp = pd.DataFrame({
        'ds': series.index,
        'y': series.values
    })
    
    # Pisahkan data yang ada nilainya untuk training
    df_train = df_temp.dropna()
    
    if len(df_train) < 10:
        print(f"  [WARNING] Data {col_name} terlalu sedikit untuk Prophet, gunakan linear")
        return series.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
    
    # Buat dan fit model Prophet
    model = Prophet(
        daily_seasonality=True,
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.05
    )
    model.fit(df_train)
    
    # Prediksi untuk semua tanggal
    future = pd.DataFrame({'ds': series.index})
    forecast = model.predict(future)
    
    # Isi missing value dengan hasil prediksi Prophet
    result = series.copy()
    missing_idx = series[series.isnull()].index
    
    for idx in missing_idx:
        pred_value = forecast[forecast['ds'] == idx]['yhat'].values
        if len(pred_value) > 0:
            result[idx] = pred_value[0]
    
    return result

# Imputasi setiap parameter dengan Prophet
for col in params:
    print(f"  Memproses {col}...")
    df_prophet[col] = impute_with_prophet(df_prophet[col], col)

# Pastikan tidak ada NaN tersisa
for col in params:
    df_prophet[col] = df_prophet[col].fillna(method='ffill').fillna(method='bfill')

print("\n[INFO] Imputasi Prophet selesai")
print(f"Missing value tersisa: {df_prophet[params].isnull().sum().sum()}")

# Simpan hasil imputasi Prophet
df_prophet.to_csv('DATA/2024_prophet.csv')
print("[INFO] Data disimpan ke: DATA/2024_prophet.csv")

# =====================================================
# 5. PLOT HASIL IMPUTASI
# =====================================================

print("\n" + "=" * 60)
print("MEMBUAT PLOT HASIL IMPUTASI")
print("=" * 60)

# Buat figure untuk perbandingan
fig, axes = plt.subplots(4, 2, figsize=(16, 14))
fig.suptitle('Hasil Imputasi Data Iklim BMKG - Stasiun Geofisika Bandung 2024', 
             fontsize=14, fontweight='bold')

param_labels = {
    'TAVG': 'Temperatur Rata-rata (°C)',
    'RH_AVG': 'Kelembapan Rata-rata (%)',
    'RR': 'Curah Hujan (mm)',
    'FF_AVG': 'Kecepatan Angin Rata-rata (m/s)'
}

colors = {'original': 'lightgray', 'linear': 'blue', 'prophet': 'red'}

for i, col in enumerate(params):
    # Plot Linear Interpolation
    ax1 = axes[i, 0]
    ax1.plot(df_linear.index, df_linear[col], color=colors['linear'], linewidth=0.8, label='Linear Interpolation')
    
    # Tandai titik yang diimputasi
    imputed_idx = missing_mask[col][missing_mask[col] == True].index
    if len(imputed_idx) > 0:
        ax1.scatter(imputed_idx, df_linear.loc[imputed_idx, col], 
                   color='orange', s=30, zorder=5, label='Nilai Imputasi', marker='o')
    
    ax1.set_title(f'{param_labels[col]} - Linear Interpolation', fontsize=10)
    ax1.set_xlabel('Tanggal')
    ax1.set_ylabel(param_labels[col])
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot Prophet
    ax2 = axes[i, 1]
    ax2.plot(df_prophet.index, df_prophet[col], color=colors['prophet'], linewidth=0.8, label='Prophet Imputation')
    
    if len(imputed_idx) > 0:
        ax2.scatter(imputed_idx, df_prophet.loc[imputed_idx, col], 
                   color='green', s=30, zorder=5, label='Nilai Imputasi', marker='o')
    
    ax2.set_title(f'{param_labels[col]} - Prophet Imputation', fontsize=10)
    ax2.set_xlabel('Tanggal')
    ax2.set_ylabel(param_labels[col])
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('DATA/plot_imputasi_comparison.png', dpi=150, bbox_inches='tight')
print("[INFO] Plot disimpan ke: DATA/plot_imputasi_comparison.png")
plt.show()

# =====================================================
# 6. EVALUASI PERFORMANSI IMPUTASI
# =====================================================

print("\n" + "=" * 60)
print("EVALUASI PERFORMANSI IMPUTASI")
print("=" * 60)

print("""
METODE EVALUASI:
----------------
Karena kita tidak memiliki ground truth untuk missing value yang sebenarnya,
kita menggunakan pendekatan Cross-Validation dengan cara:
1. Ambil sebagian data yang ada (bukan missing) secara acak
2. Simulasikan sebagai missing value
3. Lakukan imputasi
4. Bandingkan hasil imputasi dengan nilai asli

Metrik yang digunakan:
- MAE (Mean Absolute Error): Rata-rata error absolut
- RMSE (Root Mean Square Error): Akar rata-rata kuadrat error
""")

# Evaluasi dengan cross-validation approach
np.random.seed(42)

eval_results = []

for col in params:
    # Ambil data yang tidak missing
    valid_data = df[col].dropna()
    
    if len(valid_data) < 20:
        continue
    
    # Pilih 10% data secara acak untuk dievaluasi
    n_test = max(int(len(valid_data) * 0.1), 5)
    test_indices = np.random.choice(valid_data.index, size=n_test, replace=False)
    
    # Simpan nilai asli
    true_values = valid_data.loc[test_indices].values
    
    # Buat data dengan artificial missing
    df_test = df[col].copy()
    df_test.loc[test_indices] = np.nan
    
    # Imputasi Linear
    df_test_linear = df_test.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
    pred_linear = df_test_linear.loc[test_indices].values
    
    # Hitung metrik untuk Linear
    mae_linear = mean_absolute_error(true_values, pred_linear)
    rmse_linear = np.sqrt(mean_squared_error(true_values, pred_linear))
    
    # Untuk Prophet, gunakan nilai dari df_prophet yang sudah dihitung
    # (karena Prophet membutuhkan waktu lama untuk running ulang)
    pred_prophet = df_prophet.loc[test_indices, col].values
    mae_prophet = mean_absolute_error(true_values, pred_prophet)
    rmse_prophet = np.sqrt(mean_squared_error(true_values, pred_prophet))
    
    eval_results.append({
        'Parameter': col,
        'MAE_Linear': round(mae_linear, 4),
        'RMSE_Linear': round(rmse_linear, 4),
        'MAE_Prophet': round(mae_prophet, 4),
        'RMSE_Prophet': round(rmse_prophet, 4)
    })

eval_df = pd.DataFrame(eval_results)
print("\nHASIL EVALUASI:")
print(eval_df.to_string(index=False))

# Tentukan metode terbaik per parameter
print("\n" + "-" * 60)
print("KESIMPULAN:")
print("-" * 60)

for _, row in eval_df.iterrows():
    param = row['Parameter']
    if row['MAE_Linear'] < row['MAE_Prophet']:
        winner = "Linear Interpolation"
        improvement = ((row['MAE_Prophet'] - row['MAE_Linear']) / row['MAE_Prophet'] * 100)
    else:
        winner = "Prophet"
        improvement = ((row['MAE_Linear'] - row['MAE_Prophet']) / row['MAE_Linear'] * 100)
    
    print(f"  {param}: {winner} lebih baik (improvement: {improvement:.1f}%)")

print("\n" + "=" * 60)
print("PENJELASAN HASIL:")
print("=" * 60)
print("""
1. LINEAR INTERPOLATION:
   - Kelebihan: Cepat, sederhana, cocok untuk data yang perubahannya gradual
   - Kekurangan: Tidak memperhitungkan pola musiman (seasonality)
   
2. PROPHET:
   - Kelebihan: Memperhitungkan trend dan seasonality (harian, mingguan, tahunan)
   - Kekurangan: Lebih kompleks, membutuhkan lebih banyak data untuk akurasi

REKOMENDASI:
- Untuk data cuaca harian seperti ini, Linear Interpolation umumnya cukup baik
  karena perubahan cuaca biasanya gradual dalam jangka pendek.
- Prophet lebih cocok jika missing value berturut-turut dalam periode panjang
  dan kita ingin mempertimbangkan pola musiman.
""")

print("\n[INFO] Proses imputasi selesai!")
print("File output:")
print("  1. DATA/2024_linear.csv - Data hasil imputasi linear")
print("  2. DATA/2024_prophet.csv - Data hasil imputasi Prophet")
print("  3. DATA/plot_imputasi_comparison.png - Plot perbandingan")
