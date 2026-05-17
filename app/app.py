"""
app.py — WindPred Flask Application
Prediksi Kecepatan Angin BMKG Stasiun Klimatologi Jawa Barat
"""

import os, json, time, joblib, warnings
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR  = os.path.join(BASE_DIR, 'data')
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

FEATURES = ['TN','TX','TAVG','RH_AVG','RR','SS','FF_X','DDD_X',
            'day_of_year','month','FF_AVG_lag1','FF_AVG_lag7','FF_AVG_roll7',
            'time_index']

# Koordinat Stasiun Klimatologi Jawa Barat
STATION_LAT = -6.90
STATION_LON = 107.61

# Rentang dataset
DATASET_MIN_DATE = '2021-01-01'
DATASET_MAX_DATE = '2026-05-12'

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

models  = {}
scalers = {}


# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────
def load_models():
    global models, scalers
    for name, path in [('scaler_X','scaler_X.pkl'),('y','scaler_y.pkl')]:
        try:
            scalers[name] = joblib.load(os.path.join(MODEL_DIR, path))
            print(f"[OK] scaler/{name}")
        except Exception as e:
            print(f"[WARN] scaler/{name}: {e}")

    try:
        models['linear_regression'] = joblib.load(
            os.path.join(MODEL_DIR, 'linear_regression.pkl'))
        print("[OK] Linear Regression")
    except Exception as e:
        print(f"[WARN] LinReg: {e}")

    for mname, fkeras, fh5 in [
        ('ann',  'ann_model.keras',       'ann_model.h5'),
        ('lstm', 'rnn_lstm_model.keras',  'rnn_lstm_model.h5'),
    ]:
        loaded = False
        for fmt_path in [fkeras, fh5]:
            full_path = os.path.join(MODEL_DIR, fmt_path)
            if not os.path.exists(full_path):
                continue
            try:
                import tensorflow as tf
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = tf.keras.models.load_model(full_path, compile=False)
                    m.compile(optimizer='adam', loss='mse')
                # Warm-up inference agar model siap dipakai
                FEATURES_COUNT = len(FEATURES)
                _ = m.predict(np.zeros((1, FEATURES_COUNT)), verbose=0) if mname == 'ann' else None
                models[mname] = m
                print(f"[OK] {mname.upper()} (from {fmt_path})")
                loaded = True
                break
            except Exception as e:
                print(f"[WARN] {mname} failed to load {fmt_path}: {e}")
        if not loaded:
            print(f"[ERROR] {mname.upper()}: semua format gagal dimuat. Periksa file model di folder 'models/'.")

    try:
        models['kmeans']        = joblib.load(os.path.join(MODEL_DIR, 'kmeans_model.pkl'))
        models['scaler_kmeans'] = joblib.load(os.path.join(MODEL_DIR, 'scaler_kmeans.pkl'))
        print("[OK] K-Means")
    except Exception as e:
        print(f"[WARN] KMeans: {e}")

    try:
        d   = np.load(os.path.join(MODEL_DIR, 'backprop_weights.npz'))
        with open(os.path.join(MODEL_DIR, 'backprop_history.json')) as f:
            bp = json.load(f)
        ls  = bp.get('layer_sizes', [len(FEATURES), 64, 32, 1])

        class BPNet:
            def __init__(self, W, B): self.W=W; self.B=B
            def relu(self,z): return np.maximum(0,z)
            def predict(self,X):
                a=X
                for i,(W,b) in enumerate(zip(self.W,self.B)):
                    z=a@W+b; a=z if i==len(self.W)-1 else self.relu(z)
                return a.ravel()

        n = len(ls)-1
        models['backprop'] = BPNet(
            [d[f'W{i}'] for i in range(n)],
            [d[f'b{i}'] for i in range(n)])
        print("[OK] Backpropagation")
    except Exception as e:
        print(f"[WARN] Backprop: {e}")


# ─────────────────────────────────────────────────────────────
# WIND CATEGORY
# ─────────────────────────────────────────────────────────────
def wind_category(speed):
    if speed is None: return {}
    if speed < 1.5:
        return {"label":"Angin Tenang","color":"success","hex":"#22c55e","icon":"🌤",
                "beaufort":0,"beaufort_name":"Calm",
                "desc":"Kondisi sangat tenang, angin hampir tidak terasa. Asap naik hampir vertikal. Aman untuk semua aktivitas luar ruangan."}
    elif speed < 3.0:
        return {"label":"Angin Ringan","color":"info","hex":"#0ea5e9","icon":"🌬",
                "beaufort":2,"beaufort_name":"Light Breeze",
                "desc":"Angin terasa di wajah, daun-daun bergerak halus. Cocok untuk aktivitas outdoor, layangan, dan pengeringan."}
    elif speed < 5.0:
        return {"label":"Angin Sedang","color":"warning","hex":"#f59e0b","icon":"💨",
                "beaufort":3,"beaufort_name":"Gentle Breeze",
                "desc":"Daun dan ranting kecil terus bergerak. Bendera ringan berkibar. Waspadai penerbangan drone kecil."}
    else:
        return {"label":"Angin Kencang","color":"danger","hex":"#ef4444","icon":"🌪",
                "beaufort":5,"beaufort_name":"Fresh Breeze",
                "desc":"Cabang pohon besar bergerak, berjalan menentang angin terasa berat. Hindari aktivitas di ketinggian."}


# ─────────────────────────────────────────────────────────────
# PREDICT ALL MODELS
# ─────────────────────────────────────────────────────────────
def _run_model(name, X_scaled):
    if name == 'linear_regression':
        # Pipeline Ridge+Poly dilatih di X asli → inverse-transform dulu
        X_raw = scalers['scaler_X'].inverse_transform(X_scaled)
        y_pred_raw = models['linear_regression'].predict(X_raw)
        # Pipeline menghasilkan prediksi langsung di skala m/s (y asli)
        # Kembalikan sebagai array agar konsisten dengan alur inverse_transform di bawah
        # Namun karena sudah di skala asli, kita kembalikan setelah fake-scale
        return scalers['y'].transform(np.clip(y_pred_raw, 0, None).reshape(-1,1)).ravel()
    elif name == 'ann':
        return models['ann'].predict(X_scaled, verbose=0).ravel()
    elif name == 'lstm':
        with open(os.path.join(MODEL_DIR, 'lstm_seq_buffer.json')) as f:
            buf = json.load(f)
        TIMESTEPS = buf['timesteps']
        seq = np.vstack([np.array(buf['buffer']), X_scaled])[-TIMESTEPS:]
        seq = seq.reshape(1, TIMESTEPS, len(FEATURES))
        return models['lstm'].predict(seq, verbose=0).ravel()
    elif name == 'backprop':
        return models['backprop'].predict(X_scaled)
    raise ValueError(f"Unknown: {name}")


MODEL_ORDER  = ['linear_regression', 'ann', 'lstm', 'backprop']
MODEL_LABELS = {
    'linear_regression': 'Linear Regression',
    'ann':               'ANN',
    'lstm':              'RNN/LSTM',
    'backprop':          'Backpropagation'
}


def predict_all(features_dict):
    """Prediksi dari semua 4 model sekaligus."""
    if 'scaler_X' not in scalers:
        raise RuntimeError("Scaler belum dimuat. Jalankan train.py dahulu.")

    # Auto-hitung time_index jika tidak disertakan (fitur tren)
    if 'time_index' not in features_dict:
        try:
            meta = joblib.load(os.path.join(MODEL_DIR, 'time_meta.pkl'))
            min_date = pd.to_datetime(meta['min_date'])
        except Exception:
            min_date = pd.to_datetime('2021-01-01')
        features_dict = dict(features_dict)
        features_dict['time_index'] = (datetime.now() - min_date).days

    X_raw    = np.array([[features_dict[f] for f in FEATURES]], dtype=float)
    X_scaled = scalers['scaler_X'].transform(X_raw)

    results = {}
    for m in MODEL_ORDER:
        label = MODEL_LABELS[m]
        if m not in models:
            results[m] = {"label":label,"value":None,"available":False,"error":"Model belum dimuat"}
            continue
        try:
            y_s   = _run_model(m, X_scaled)
            y_val = float(scalers['y'].inverse_transform(
                np.array(y_s).reshape(-1,1)).ravel()[0])
            y_val = max(0.0, round(y_val, 4))
            results[m] = {"label":label,"value":y_val,"available":True,
                          "category":wind_category(y_val)}
        except Exception as e:
            results[m] = {"label":label,"value":None,"available":False,"error":str(e)}

    vals     = [r['value'] for r in results.values() if r.get('available') and r['value'] is not None]
    ensemble = round(float(np.mean(vals)), 4) if vals else None

    try:
        with open(os.path.join(MODEL_DIR,'model_metrics.json')) as f:
            best = json.load(f).get('best_model','linear_regression')
    except Exception:
        best = 'linear_regression'

    return {
        "models":       results,
        "model_order":  MODEL_ORDER,
        "ensemble":     ensemble,
        "ensemble_cat": wind_category(ensemble),
        "best_model":   best,
        "best_value":   results.get(best, {}).get('value'),
        "timestamp":    datetime.now().isoformat()
    }


# ─────────────────────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────────────────────
def get_historical_data(n=90):
    try:
        df = pd.read_csv(os.path.join(DATA_DIR,'bmkg_preprocessed.csv'))
        df = df.tail(n)[['TANGGAL','FF_AVG','FF_X','TAVG','RH_AVG','RR','SS']].copy()
        df['TANGGAL'] = pd.to_datetime(df['TANGGAL']).dt.strftime('%Y-%m-%d')
        return df.to_dict(orient='records')
    except Exception:
        return []


def get_default_inputs():
    try:
        df   = pd.read_csv(os.path.join(DATA_DIR,'bmkg_preprocessed.csv'))
        last = df.iloc[-1]
        return {f: round(float(last[f]),2) for f in FEATURES if f in last.index}
    except Exception:
        return {}


def _get_from_dataset(date_str):
    """Cari data di dataset historis. Return (row_dict, found, nearest_date)."""
    df = pd.read_csv(os.path.join(DATA_DIR,'bmkg_preprocessed.csv'))
    df['TANGGAL'] = pd.to_datetime(df['TANGGAL'])
    target = pd.to_datetime(date_str)

    row = df[df['TANGGAL'] == target]
    if not row.empty:
        r = row.iloc[0]
        return {f: round(float(r[f]),4) for f in FEATURES if f in r.index}, True, date_str

    before = df[df['TANGGAL'] < target].tail(1)
    if not before.empty:
        r = before.iloc[0]
        return {f: round(float(r[f]),4) for f in FEATURES if f in r.index}, False, r['TANGGAL'].strftime('%Y-%m-%d')

    return None, False, None


def _fetch_openmeteo(date_str):
    """
    Ambil data cuaca dari Open-Meteo untuk tanggal tertentu.

    Strategi endpoint (berurutan, coba berikutnya jika gagal):
      1. forecast + past_days  → untuk tanggal dalam ±92 hari dari hari ini
                                  (menangkap "recent past" yang tidak ada di archive)
      2. archive (ERA5)        → untuk data historis > 5 hari lalu
      3. archive-api subdomain → fallback archive alternatif

    Return (dict_fitur, None) jika berhasil, atau (None, pesan_error) jika gagal.
    """
    target    = pd.to_datetime(date_str)
    today     = pd.Timestamp.now().normalize()
    delta_days = (today - target).days   # positif = masa lalu, negatif = masa depan

    DAILY_VARS = ('temperature_2m_max,temperature_2m_min,temperature_2m_mean,'
                  'precipitation_sum,sunshine_duration,'
                  'windspeed_10m_max,winddirection_10m_dominant')

    BASE_PARAMS = {
        'latitude': STATION_LAT, 'longitude': STATION_LON,
        'daily': DAILY_VARS, 'timezone': 'Asia/Jakarta', 'windspeed_unit': 'ms'
    }

    # Daftar strategi endpoint yang akan dicoba berurutan
    attempts = []

    if delta_days >= 0:
        # Tanggal masa lalu atau hari ini
        past_days_needed = max(delta_days + 1, 1)
        if past_days_needed <= 92:
            # Forecast endpoint dengan past_days — lebih reliabel untuk recent data
            attempts.append(('forecast_past', 'https://api.open-meteo.com/v1/forecast', {
                **BASE_PARAMS,
                'past_days': min(past_days_needed, 92),
                'forecast_days': 1,
            }))
        # Archive ERA5 — kadang ada delay ~5 hari, tapi coba tetap
        attempts.append(('archive', 'https://api.open-meteo.com/v1/archive', {
            **BASE_PARAMS, 'start_date': date_str, 'end_date': date_str,
        }))
        # Archive subdomain alternatif
        attempts.append(('archive_sub', 'https://archive-api.open-meteo.com/v1/archive', {
            **BASE_PARAMS, 'start_date': date_str, 'end_date': date_str,
        }))
    else:
        # Tanggal masa depan (future)
        forecast_days_needed = min(abs(delta_days) + 1, 16)
        attempts.append(('forecast_future', 'https://api.open-meteo.com/v1/forecast', {
            **BASE_PARAMS, 'forecast_days': forecast_days_needed,
        }))

    last_err = "Semua endpoint gagal"
    for attempt_name, url, params in attempts:
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 404:
                last_err = f"{attempt_name}: 404 Not Found (tanggal mungkin di luar rentang endpoint ini)"
                continue
            r.raise_for_status()
            d = r.json()

            if d.get('error'):
                last_err = f"{attempt_name}: {d.get('reason','API error')}"
                continue

            daily = d.get('daily', {})
            dates = daily.get('time', [])

            # Cari index tanggal yang diinginkan
            idx = None
            for i, dt in enumerate(dates):
                if dt == date_str:
                    idx = i
                    break
            if idx is None and len(dates) > 0:
                idx = 0  # ambil data pertama jika tidak ditemukan exact match

            if idx is None:
                last_err = f"{attempt_name}: tanggal {date_str} tidak ada di response"
                continue

            def get_val(key, default=0.0):
                vals = daily.get(key, [])
                v = vals[idx] if idx < len(vals) else None
                return float(v) if v is not None else default

            TX     = get_val('temperature_2m_max', 30.0)
            TN     = get_val('temperature_2m_min', 22.0)
            TAVG   = get_val('temperature_2m_mean', (TX+TN)/2)
            RR     = get_val('precipitation_sum',   0.0)
            SS     = round(get_val('sunshine_duration', 18000) / 3600, 1)
            FF_X   = get_val('windspeed_10m_max',   5.0)
            DDD_X  = get_val('winddirection_10m_dominant', 180.0)
            RH_AVG = max(50.0, min(100.0, 90.0 - (TX - 28) * 2))

            # Lag features dari dataset terakhir sebagai proxy
            try:
                df_hist = pd.read_csv(os.path.join(DATA_DIR, 'bmkg_preprocessed.csv'))
                last    = df_hist.iloc[-1]
                lag1    = float(last['FF_AVG_lag1'])
                lag7    = float(last['FF_AVG_lag7'])
                roll7   = float(last['FF_AVG_roll7'])
            except Exception:
                lag1 = lag7 = roll7 = 2.0

            dt_obj      = pd.to_datetime(date_str)
            day_of_year = int(dt_obj.strftime('%j'))
            month       = dt_obj.month

            feats = {
                'TN': round(TN,2), 'TX': round(TX,2), 'TAVG': round(TAVG,2),
                'RH_AVG': round(RH_AVG,2), 'RR': round(RR,2), 'SS': round(SS,2),
                'FF_X': round(FF_X,2), 'DDD_X': round(DDD_X,2),
                'day_of_year': day_of_year, 'month': month,
                'FF_AVG_lag1': round(lag1,4), 'FF_AVG_lag7': round(lag7,4),
                'FF_AVG_roll7': round(roll7,4)
            }
            return feats, None

        except requests.exceptions.Timeout:
            last_err = f"{attempt_name}: timeout (>10 detik)"
        except requests.exceptions.ConnectionError:
            last_err = f"{attempt_name}: koneksi gagal"
        except Exception as e:
            last_err = f"{attempt_name}: {e}"

    return None, last_err


def get_inputs_for_date(date_str):
    """
    Cari atau estimasi input fitur untuk tanggal tertentu.
    Prioritas: 1) Dataset BMKG lokal (exact match)
               2) Open-Meteo API — archive atau forecast (tanggal apa pun)
               3) Fallback data terdekat dari dataset lokal (jika API gagal)
    """
    target     = pd.to_datetime(date_str)
    ds_min     = pd.to_datetime(DATASET_MIN_DATE)
    ds_max     = pd.to_datetime(DATASET_MAX_DATE)
    in_dataset = ds_min <= target <= ds_max

    # 1. Dataset BMKG lokal — hanya jika tanggal dalam rentang dataset
    if in_dataset:
        try:
            feats, found, nearest = _get_from_dataset(date_str)
            if feats and found:
                return {
                    "found": True, "tanggal": date_str,
                    "source": "dataset_bmkg",
                    "source_label": "Dataset BMKG Lokal",
                    "data": feats
                }
            # Jika dalam rentang tapi tidak exact (misal data hilang), tetap lanjut ke API
        except Exception:
            pass

    # 2. Open-Meteo API — berlaku untuk tanggal berapa pun:
    #    - Tanggal historis (sebelum hari ini): endpoint archive
    #    - Tanggal hari ini / masa depan s.d 16 hari: endpoint forecast
    #    Endpoint Open-Meteo gratis, tidak perlu API key.
    feats_api, api_err = _fetch_openmeteo(date_str)
    if feats_api:
        today     = pd.Timestamp.now().normalize()
        is_future = target > today
        return {
            "found":        True,
            "tanggal":      date_str,
            "source":       "open_meteo_forecast" if is_future else "open_meteo",
            "source_label": "Open-Meteo Forecast API" if is_future else "Open-Meteo Historical Weather API",
            "note":         "Fitur RH_AVG dan lag diestimasi dari pola historis terakhir.",
            "data":         feats_api
        }

    # 3. Fallback terakhir — data terdekat dari dataset lokal
    #    Ini terjadi hanya jika Open-Meteo tidak dapat diakses (offline / timeout).
    #    Prediksi masih bisa dijalankan, namun hasilnya kurang akurat karena
    #    menggunakan data observasi dari tanggal yang berbeda.
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, 'bmkg_preprocessed.csv'))
        df['TANGGAL'] = pd.to_datetime(df['TANGGAL'])
        before = df[df['TANGGAL'] <= target].tail(1)
        if not before.empty:
            r       = before.iloc[0]
            nearest = r['TANGGAL'].strftime('%Y-%m-%d')
            feats   = {f: round(float(r[f]), 4) for f in FEATURES if f in r.index}
            return {
                "found":        False,
                "tanggal":      date_str,
                "nearest":      nearest,
                "source":       "nearest_historical",
                "source_label": f"Data historis terdekat ({nearest})",
                "api_error":    api_err,
                "warning":      (
                    f"Open-Meteo API tidak dapat diakses ({api_err}). "
                    f"Prediksi menggunakan data observasi terdekat ({nearest}) "
                    f"sebagai estimasi. Hasil prediksi mungkin kurang presisi."
                ),
                "data": feats
            }
    except Exception as e:
        return {"found": False, "error": str(e)}

    return {"found": False, "error": "Tidak ada data tersedia untuk tanggal ini"}


def _cache_get(key, ttl=3600):
    p = os.path.join(CACHE_DIR, key+'.json')
    if os.path.exists(p) and (time.time()-os.path.getmtime(p)) < ttl:
        with open(p) as f: return json.load(f)
    return None

def _cache_set(key, data):
    with open(os.path.join(CACHE_DIR, key+'.json'),'w') as f: json.dump(data,f)


def fetch_weather_forecast():
    """Prakiraan 7 hari dari Open-Meteo. Cache 1 jam. Fallback ke data historis."""
    cached = _cache_get('forecast')
    if cached: return cached

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        'latitude': STATION_LAT, 'longitude': STATION_LON,
        'daily': ('windspeed_10m_max,windgusts_10m_max,temperature_2m_max,'
                  'temperature_2m_min,precipitation_sum,sunshine_duration'),
        'current_weather': 'true', 'timezone': 'Asia/Jakarta',
        'forecast_days': 7, 'windspeed_unit': 'ms'
    }
    try:
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        d = r.json()
        daily = d.get('daily',{}); dates = daily.get('time',[])
        forecasts = [{
            "date": dates[i],
            "wind_max":   daily.get('windspeed_10m_max',[None]*7)[i],
            "gust_max":   daily.get('windgusts_10m_max',[None]*7)[i],
            "temp_max":   daily.get('temperature_2m_max',[None]*7)[i],
            "temp_min":   daily.get('temperature_2m_min',[None]*7)[i],
            "rain":       daily.get('precipitation_sum',[None]*7)[i],
            "sunshine_h": round(daily.get('sunshine_duration',[0]*7)[i]/3600,1)
        } for i in range(len(dates))]
        cw = d.get('current_weather',{})
        result = {
            "source":"Open-Meteo (WMO ERA5)","station":"Bandung, Jawa Barat",
            "current":{"windspeed":cw.get('windspeed'),"winddirection":cw.get('winddirection'),
                       "temperature":cw.get('temperature'),"time":cw.get('time')},
            "forecasts":forecasts,"fetched_at":datetime.now().isoformat(),"error":None
        }
        _cache_set('forecast', result)
        return result
    except Exception as e:
        return _fallback_forecast(str(e))


def _fallback_forecast(err=""):
    try:
        df   = pd.read_csv(os.path.join(DATA_DIR,'bmkg_preprocessed.csv'))
        last = df.tail(7)
        forecasts=[{
            "date": str(r['TANGGAL'])[:10],
            "wind_max":float(r.get('FF_X',0)),"gust_max":None,
            "temp_max":float(r.get('TX',0)),"temp_min":float(r.get('TN',0)),
            "rain":float(r.get('RR',0)),"sunshine_h":float(r.get('SS',0))
        } for _,r in last.iterrows()]
        lr = last.iloc[-1]
        return {
            "source":"Data Historis BMKG (Mode Offline)","station":"Stasiun Klimatologi Jawa Barat",
            "current":{"windspeed":float(lr['FF_AVG']),"winddirection":float(lr['DDD_X']),
                       "temperature":float(lr['TAVG']),"time":str(lr['TANGGAL'])[:10]},
            "forecasts":forecasts,"fetched_at":datetime.now().isoformat(),
            "error":f"Layanan cuaca eksternal tidak tersedia. Menampilkan 7 data historis terakhir dari dataset BMKG."
        }
    except Exception as e2:
        return {"source":"unavailable","forecasts":[],"error":str(e2),"fetched_at":datetime.now().isoformat()}


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────
@app.route('/')
def index():
    metrics = {}
    try:
        with open(os.path.join(MODEL_DIR,'model_metrics.json')) as f: metrics = json.load(f)
    except Exception: pass
    return render_template('index.html', metrics=metrics, historical=get_historical_data(90))


@app.route('/predict')
def predict_page():
    return render_template('predict.html', defaults=get_default_inputs(), features=FEATURES)


@app.route('/compare')
def compare_page():
    metrics=ann_hist=lstm_hist=bp_hist={}
    try:
        with open(os.path.join(MODEL_DIR,'model_metrics.json')) as f: metrics=json.load(f)
        with open(os.path.join(MODEL_DIR,'ann_history.json'))    as f: ann_hist=json.load(f)
        with open(os.path.join(MODEL_DIR,'lstm_history.json'))   as f: lstm_hist=json.load(f)
        with open(os.path.join(MODEL_DIR,'backprop_history.json')) as f: bp_hist=json.load(f)
    except Exception: pass
    return render_template('compare.html', metrics=metrics,
                           ann_hist=ann_hist, lstm_hist=lstm_hist, bp_hist=bp_hist)


@app.route('/clustering')
def clustering_page():
    km={}
    try:
        with open(os.path.join(MODEL_DIR,'kmeans_metrics.json')) as f: km=json.load(f)
    except Exception: pass
    return render_template('clustering.html', kmeans_metrics=km)


@app.route('/dataset')
def dataset_page():
    return render_template('dataset.html')


@app.route('/about')
def about_page():
    metrics={}
    try:
        with open(os.path.join(MODEL_DIR,'model_metrics.json')) as f: metrics=json.load(f)
    except Exception: pass
    return render_template('about.html', metrics=metrics)


# ─────────────────────────────────────────────────────────────
# API ENDPOINTS — JSON REST API
# ─────────────────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json(force=True)
        if not data: return jsonify({"error":"Body JSON kosong"}),400
        # time_index boleh tidak disertakan — akan dihitung otomatis oleh predict_all
        required = [f for f in FEATURES if f != 'time_index']
        missing = [f for f in required if f not in data]
        if missing: return jsonify({"error":f"Fitur kurang: {missing}"}),400
        features = {f:float(data[f]) for f in FEATURES if f in data}
        return jsonify(predict_all(features))
    except Exception as e:
        return jsonify({"error":str(e)}),500


@app.route('/api/predict-by-date', methods=['POST'])
def api_predict_by_date():
    try:
        data     = request.get_json(force=True)
        date_str = data.get('date','')
        if not date_str: return jsonify({"error":"Field 'date' wajib"}),400

        date_info = get_inputs_for_date(date_str)
        if not date_info.get('data'):
            return jsonify({"error": date_info.get('error','Data tidak tersedia')}),404

        result = predict_all(date_info['data'])
        result['date_info']     = date_info
        result['features_used'] = date_info['data']
        return jsonify(result)
    except Exception as e:
        return jsonify({"error":str(e)}),500


@app.route('/api/metrics')
def api_metrics():
    try:
        with open(os.path.join(MODEL_DIR,'model_metrics.json')) as f: return jsonify(json.load(f))
    except Exception as e: return jsonify({"error":str(e)}),500


@app.route('/api/cluster-data')
def api_cluster_data():
    try:
        df = pd.read_csv(os.path.join(DATA_DIR,'clustered_data.csv'))
        df['TANGGAL'] = pd.to_datetime(df['TANGGAL']).dt.strftime('%Y-%m-%d')
        return jsonify(df.to_dict(orient='records'))
    except Exception as e: return jsonify({"error":str(e)}),500


@app.route('/api/historical')
def api_historical():
    return jsonify(get_historical_data(request.args.get('n',90,type=int)))


@app.route('/api/dataset')
def api_dataset():
    try:
        page=request.args.get('page',1,type=int)
        per_page=request.args.get('per_page',50,type=int)
        search=request.args.get('search','',type=str)
        sort_by=request.args.get('sort_by','TANGGAL',type=str)
        order=request.args.get('order','desc',type=str)

        df = pd.read_csv(os.path.join(DATA_DIR,'bmkg_merged.csv'))
        df['TANGGAL'] = pd.to_datetime(df['TANGGAL']).dt.strftime('%Y-%m-%d')
        cols = ['TANGGAL','TN','TX','TAVG','RH_AVG','RR','SS','FF_X','DDD_X','FF_AVG','DDD_CAR']
        df   = df[[c for c in cols if c in df.columns]]
        if search: df = df[df['TANGGAL'].str.contains(search,na=False)]
        if sort_by in df.columns: df = df.sort_values(sort_by,ascending=(order=='asc'))
        total=len(df); start=(page-1)*per_page
        paged=df.iloc[start:start+per_page]
        return jsonify({"data":paged.fillna('—').to_dict(orient='records'),
                        "total":total,"page":page,"per_page":per_page,
                        "total_pages":(total+per_page-1)//per_page})
    except Exception as e: return jsonify({"error":str(e)}),500


@app.route('/api/model-history/<model_name>')
def api_model_history(model_name):
    fm={'ann':'ann_history.json','lstm':'lstm_history.json','backprop':'backprop_history.json'}
    if model_name not in fm: return jsonify({"error":"Tidak ditemukan"}),404
    try:
        with open(os.path.join(MODEL_DIR,fm[model_name])) as f: return jsonify(json.load(f))
    except Exception as e: return jsonify({"error":str(e)}),500


@app.route('/api/kmeans-metrics')
def api_kmeans_metrics():
    try:
        with open(os.path.join(MODEL_DIR,'kmeans_metrics.json')) as f: return jsonify(json.load(f))
    except Exception as e: return jsonify({"error":str(e)}),500


@app.route('/api/weather-forecast')
def api_weather_forecast():
    return jsonify(fetch_weather_forecast())


@app.route('/api/date-lookup')
def api_date_lookup():
    date_str = request.args.get('date','')
    if not date_str: return jsonify({"error":"Parameter 'date' wajib"}),400
    return jsonify(get_inputs_for_date(date_str))


@app.route('/api/default-inputs')
def api_default_inputs():
    return jsonify(get_default_inputs())


@app.route('/api/model-status')
def api_model_status():
    """Cek status setiap model — berguna untuk debug apakah model berhasil dimuat."""
    status = {}
    for m in MODEL_ORDER:
        loaded = m in models
        status[m] = {
            "label":  MODEL_LABELS[m],
            "loaded": loaded,
            "type":   type(models[m]).__name__ if loaded else None
        }
    status['scalers'] = {k: k in scalers for k in ['scaler_X', 'y']}
    return jsonify(status)


@app.route('/api/reload-models', methods=['POST'])
def api_reload_models():
    """Reload semua model tanpa restart server. Panggil jika model gagal dimuat saat startup."""
    try:
        load_models()
        loaded = [m for m in MODEL_ORDER if m in models]
        failed = [m for m in MODEL_ORDER if m not in models]
        return jsonify({
            "success": True,
            "loaded":  loaded,
            "failed":  failed,
            "message": f"{len(loaded)}/{len(MODEL_ORDER)} model berhasil dimuat."
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'): return jsonify({"error":"Endpoint tidak ditemukan"}),404
    return render_template('404.html'),404

@app.errorhandler(500)
def server_error(e): return jsonify({"error":"Internal server error"}),500


with app.app_context():
    load_models()

if __name__ == '__main__':
    port = int(os.environ.get('PORT',5000))
    app.run(debug=True, host='0.0.0.0', port=port)
