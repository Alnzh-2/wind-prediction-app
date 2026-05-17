"""
download_era5_merge.py
======================
Jalankan script ini SATU KALI dari folder windpred_fixed/:

    python download_era5_merge.py

Yang dilakukan:
  1. Download data ERA5 dari Open-Meteo (Jan 2021 – 13 Mei 2024) — gratis, CC BY 4.0
  2. Gabung dengan dataset BMKG lokal (bmkg_merged.csv)
  3. Simpan ke data/bmkg_merged.csv (overwrite) + backup CSV lama
  4. Update otomatis DATASET_MIN_DATE di app/app.py
  5. Update otomatis semua label di templates (index.html, predict.html, about.html)

Setelah selesai, restart Flask dan web akan menampilkan ~1.958 baris.
"""

import os, sys, json, shutil, re
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# ── KONFIGURASI ───────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, 'data')
APP_PY       = os.path.join(BASE_DIR, 'app', 'app.py')
INDEX_HTML   = os.path.join(BASE_DIR, 'app', 'templates', 'index.html')
PREDICT_HTML = os.path.join(BASE_DIR, 'app', 'templates', 'predict.html')
ABOUT_HTML   = os.path.join(BASE_DIR, 'app', 'templates', 'about.html')
BMKG_CSV     = os.path.join(DATA_DIR, 'bmkg_merged.csv')
OUTPUT_CSV   = os.path.join(DATA_DIR, 'bmkg_merged.csv')   # overwrite

# Stasiun Klimatologi Jawa Barat
LAT, LON = -6.90, 107.61

# ERA5: ambil dari 2021-01-01 sampai 1 hari sebelum dataset BMKG
ERA5_START = '2021-01-01'
ERA5_END   = '2024-05-13'   # bmkg_merged mulai 2024-05-14


def download_era5(lat, lon, start_date, end_date):
    """
    Ambil data cuaca harian dari Open-Meteo Historical Weather API (ERA5).
    Lisensi data: CC BY 4.0 — Copernicus Climate Change Service / ECMWF
    Referensi   : Hersbach et al. (2020). doi:10.1002/qj.3803
    """
    # Bagi per tahun supaya tidak timeout
    chunks = []
    start = pd.to_datetime(start_date)
    end   = pd.to_datetime(end_date)

    current = start
    while current <= end:
        chunk_end = min(current.replace(month=12, day=31), end)
        print(f'  Mengambil {current.date()} → {chunk_end.date()} ...', end=' ', flush=True)

        url = 'https://archive-api.open-meteo.com/v1/archive'
        params = {
            'latitude':  lat,
            'longitude': lon,
            'start_date': current.strftime('%Y-%m-%d'),
            'end_date':   chunk_end.strftime('%Y-%m-%d'),
            'daily': ','.join([
                'temperature_2m_max',
                'temperature_2m_min',
                'temperature_2m_mean',
                'relative_humidity_2m_mean',
                'precipitation_sum',
                'sunshine_duration',
                'windspeed_10m_max',
                'winddirection_10m_dominant',
                'windspeed_10m_mean',
            ]),
            'timezone': 'Asia/Jakarta',
            'windspeed_unit': 'ms'
        }

        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            d = r.json()
            if d.get('error'):
                print(f'ERROR: {d.get("reason")}')
                return None
        except Exception as e:
            print(f'GAGAL: {e}')
            return None

        daily = d['daily']
        df_chunk = pd.DataFrame({
            'TANGGAL': daily['time'],
            'TN':      daily['temperature_2m_min'],
            'TX':      daily['temperature_2m_max'],
            'TAVG':    daily['temperature_2m_mean'],
            'RH_AVG':  daily['relative_humidity_2m_mean'],
            'RR':      daily['precipitation_sum'],
            'SS':      [round(v / 3600, 1) if v else 0.0 for v in daily['sunshine_duration']],
            'FF_X':    daily['windspeed_10m_max'],
            'DDD_X':   daily['winddirection_10m_dominant'],
            'FF_AVG':  daily['windspeed_10m_mean'],
            'DDD_CAR': 'ERA5',
        })
        chunks.append(df_chunk)
        print(f'{len(df_chunk)} baris OK')
        current = chunk_end + pd.Timedelta(days=1)

    return pd.concat(chunks, ignore_index=True) if chunks else None


def update_file_text(path, replacements):
    """Lakukan serangkaian penggantian teks di sebuah file."""
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    for old, new in replacements:
        content = content.replace(old, new)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def main():
    print('=' * 60)
    print('WindPred — ERA5 Data Merger')
    print('=' * 60)

    # ── 1. Backup CSV lama ──────────────────────────────────────
    backup_path = BMKG_CSV.replace('.csv', '_bmkg_only_backup.csv')
    if not os.path.exists(backup_path):
        shutil.copy(BMKG_CSV, backup_path)
        print(f'[OK] Backup disimpan: {os.path.basename(backup_path)}')

    # ── 2. Baca dataset BMKG ────────────────────────────────────
    df_bmkg = pd.read_csv(BMKG_CSV)
    df_bmkg['TANGGAL'] = pd.to_datetime(df_bmkg['TANGGAL'])
    bmkg_start = df_bmkg['TANGGAL'].min()
    bmkg_end   = df_bmkg['TANGGAL'].max()
    print(f'\n[INFO] Dataset BMKG: {len(df_bmkg)} baris')
    print(f'       Rentang: {bmkg_start.date()} → {bmkg_end.date()}')

    # ── 3. Download ERA5 ────────────────────────────────────────
    print(f'\n[DOWNLOAD] ERA5 Open-Meteo: {ERA5_START} → {ERA5_END}')
    df_era5 = download_era5(LAT, LON, ERA5_START, ERA5_END)

    if df_era5 is None:
        print('\n[ERROR] Gagal download ERA5. Cek koneksi internet dan coba lagi.')
        sys.exit(1)

    df_era5['TANGGAL'] = pd.to_datetime(df_era5['TANGGAL'])
    print(f'\n[OK] ERA5: {len(df_era5)} baris berhasil diunduh')

    # ── 4. Merge & dedup ────────────────────────────────────────
    COLS = ['TANGGAL','TN','TX','TAVG','RH_AVG','RR','SS','FF_X','DDD_X','FF_AVG','DDD_CAR']
    for col in COLS:
        if col not in df_bmkg.columns:
            df_bmkg[col] = None
        if col not in df_era5.columns:
            df_era5[col] = None

    df_merged = pd.concat([df_era5[COLS], df_bmkg[COLS]], ignore_index=True)
    df_merged = df_merged.sort_values('TANGGAL').drop_duplicates('TANGGAL').reset_index(drop=True)

    # Interpolasi missing values
    num_cols = ['TN','TX','TAVG','RH_AVG','RR','SS','FF_X','DDD_X','FF_AVG']
    for col in num_cols:
        df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')
    df_merged[num_cols] = df_merged[num_cols].interpolate(method='linear', limit_direction='both')

    new_start = df_merged['TANGGAL'].min()
    new_end   = df_merged['TANGGAL'].max()
    total     = len(df_merged)

    print(f'\n[MERGE]')
    print(f'  Dataset lama : {len(df_bmkg):,} baris')
    print(f'  ERA5 baru    : {len(df_era5):,} baris')
    print(f'  Total gabungan: {total:,} baris')
    print(f'  Rentang baru : {new_start.date()} → {new_end.date()}')

    # ── 5. Simpan CSV ───────────────────────────────────────────
    df_merged.to_csv(OUTPUT_CSV, index=False)
    print(f'\n[OK] Disimpan: {OUTPUT_CSV}')

    # Juga simpan ke bmkg_preprocessed.csv (dipakai app.py untuk lookup)
    # Tambah feature engineering columns
    df_pp = df_merged.copy()
    df_pp['day_of_year']   = df_pp['TANGGAL'].dt.dayofyear
    df_pp['month']         = df_pp['TANGGAL'].dt.month
    df_pp['FF_AVG_lag1']   = df_pp['FF_AVG'].shift(1)
    df_pp['FF_AVG_lag7']   = df_pp['FF_AVG'].shift(7)
    df_pp['FF_AVG_roll7']  = df_pp['FF_AVG'].shift(1).rolling(7).mean()
    df_pp = df_pp.dropna(subset=['FF_AVG_lag1','FF_AVG_lag7','FF_AVG_roll7'])
    df_pp.to_csv(os.path.join(DATA_DIR, 'bmkg_preprocessed.csv'), index=False)
    print(f'[OK] Preprocessed disimpan ({len(df_pp):,} baris)')

    # ── 6. Update app.py ────────────────────────────────────────
    new_min_str = new_start.strftime('%Y-%m-%d')
    new_max_str = new_end.strftime('%Y-%m-%d')
    update_file_text(APP_PY, [
        ("DATASET_MIN_DATE = '2024-05-14'", f"DATASET_MIN_DATE = '{new_min_str}'"),
        ("DATASET_MIN_DATE = '2021-01-01'", f"DATASET_MIN_DATE = '{new_min_str}'"),  # idempotent
    ])
    print(f'\n[OK] app.py: DATASET_MIN_DATE → {new_min_str}')

    # ── 7. Update label di template ─────────────────────────────
    new_min_label = new_start.strftime('%#d %b %Y').replace('Jan','Jan').replace('May','Mei') \
                    .replace('Feb','Feb').replace('Mar','Mar').replace('Apr','Apr') \
                    .replace('Jun','Jun').replace('Jul','Jul').replace('Aug','Ags') \
                    .replace('Sep','Sep').replace('Oct','Okt').replace('Nov','Nov').replace('Dec','Des')
    new_max_label = new_end.strftime('%#d %b %Y') \
                    .replace('May','Mei').replace('Aug','Ags').replace('Oct','Okt').replace('Dec','Des')

    # Format: "1 Jan 2021 – 12 Mei 2026"
    range_label = f'{new_min_label} – {new_max_label}'

    # index.html: stat card baris data
    update_file_text(INDEX_HTML, [
        ('<div class="stat-value">729</div>',  f'<div class="stat-value">{total:,}</div>'),
        ('<div class="stat-value">1,958</div>', f'<div class="stat-value">{total:,}</div>'),  # idempotent
    ])
    print(f'[OK] index.html: stat-value → {total:,}')

    # predict.html: label rentang dataset
    update_file_text(PREDICT_HTML, [
        ('14 Mei 2024 – 12 Mei 2026', range_label),
        ('1 Jan 2021 – 12 Mei 2026',  range_label),  # idempotent
    ])
    print(f'[OK] predict.html: rentang → {range_label}')

    # about.html: Rentang Data + Total Data
    update_file_text(ABOUT_HTML, [
        ("('Rentang Data','14 Mei 2024 – 12 Mei 2026')",
         f"('Rentang Data','{range_label}')"),
        (f"('Rentang Data','1 Jan 2021 – 12 Mei 2026')",
         f"('Rentang Data','{range_label}')"),  # idempotent
        ("('Total Data','729 hari observasi')",
         f"('Total Data','{total:,} hari observasi (BMKG + ERA5)')"),
        (f"('Total Data','{total:,} hari observasi (BMKG + ERA5)')",
         f"('Total Data','{total:,} hari observasi (BMKG + ERA5)')"),  # idempotent
    ])
    print(f'[OK] about.html: Total Data → {total:,} hari')

    # ── 8. Ringkasan ────────────────────────────────────────────
    print()
    print('=' * 60)
    print('SELESAI!')
    print('=' * 60)
    print(f'  Total baris dataset : {total:,}')
    print(f'  Rentang             : {new_start.date()} → {new_end.date()}')
    print(f'  Sumber ERA5         : Open-Meteo / Copernicus ERA5 (CC BY 4.0)')
    print()
    print('Langkah selanjutnya:')
    print('  1. Jalankan: python train.py   (retrain semua model dengan data baru)')
    print('  2. Jalankan: python app/app.py (restart Flask)')
    print('  3. Web akan menampilkan dataset 1.900+ baris.')
    print()
    print('Catatan untuk laporan:')
    print('  Sitasi ERA5: Hersbach, H., et al. (2020). The ERA5 global reanalysis.')
    print('  Quarterly Journal of the Royal Meteorological Society, 146(730), 1999-2049.')
    print('  https://doi.org/10.1002/qj.3803')
    print('  Data diakses via Open-Meteo API (CC BY 4.0): https://open-meteo.com')


if __name__ == '__main__':
    main()
