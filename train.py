"""
train.py - Script pelatihan model prediksi kecepatan angin BMKG
Stasiun Klimatologi Jawa Barat (ID WMO: 96753)

PERBAIKAN v8:
1. Tambah fitur 'time_index' (linear trend) agar model bisa menangkap
   tren angin yang meningkat dari tahun ke tahun → R² meningkat signifikan
2. Linear Regression: Ridge + Polynomial(deg=2) + fitur trend → R² positif
3. LSTM: safe_mape() menghilangkan nilai MAPE jutaan persen
4. Backpropagation: Adam optimizer + L2 + arsitektur ramping
5. Evaluasi konsisten di skala asli (m/s)
"""

import os, json, time, warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             silhouette_score, r2_score)

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# PERBAIKAN: tambah 'time_index' sebagai fitur tren
FEATURES = ['TN','TX','TAVG','RH_AVG','RR','SS','FF_X','DDD_X',
            'day_of_year','month','FF_AVG_lag1','FF_AVG_lag7','FF_AVG_roll7',
            'time_index']
TARGET = 'FF_AVG'
MAPE_THRESHOLD = 0.1


# ============================================================
# BAGIAN 1 — PREPROCESSING
# ============================================================

def load_and_preprocess():
    print("\n[1/7] PREPROCESSING DATA...")
    df = pd.read_csv(os.path.join(DATA_DIR, 'bmkg_merged.csv'))
    df['TANGGAL'] = pd.to_datetime(df['TANGGAL'])
    df = df.sort_values('TANGGAL').reset_index(drop=True)

    num_cols = ['TN','TX','TAVG','RH_AVG','RR','SS','FF_X','DDD_X','FF_AVG']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[num_cols] = df[num_cols].interpolate(method='linear', limit_direction='both')

    df['day_of_year']  = df['TANGGAL'].dt.dayofyear
    df['month']        = df['TANGGAL'].dt.month
    df['FF_AVG_lag1']  = df['FF_AVG'].shift(1)
    df['FF_AVG_lag7']  = df['FF_AVG'].shift(7)
    df['FF_AVG_roll7'] = df['FF_AVG'].shift(1).rolling(window=7).mean()
    # PERBAIKAN: fitur tren waktu linear (hari ke-N sejak awal dataset)
    df['time_index']   = (df['TANGGAL'] - df['TANGGAL'].min()).dt.days

    df = df.dropna(subset=FEATURES+[TARGET]).reset_index(drop=True)
    df.to_csv(os.path.join(DATA_DIR, 'bmkg_preprocessed.csv'), index=False)
    print(f"    Total data: {len(df)} baris | {df['TANGGAL'].min().date()} → {df['TANGGAL'].max().date()}")

    n = len(df)
    n_train = int(n*0.70); n_val = int(n*0.15)
    df_train = df.iloc[:n_train]
    df_val   = df.iloc[n_train:n_train+n_val]
    df_test  = df.iloc[n_train+n_val:]
    print(f"    Split: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")
    print(f"    Train FF_AVG mean={df_train[TARGET].mean():.3f} | Test FF_AVG mean={df_test[TARGET].mean():.3f}")

    X_train = df_train[FEATURES].values; y_train = df_train[TARGET].values
    X_val   = df_val[FEATURES].values;   y_val   = df_val[TARGET].values
    X_test  = df_test[FEATURES].values;  y_test  = df_test[TARGET].values

    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_val_s   = scaler_X.transform(X_val)
    X_test_s  = scaler_X.transform(X_test)

    scaler_y = MinMaxScaler()
    y_train_s = scaler_y.fit_transform(y_train.reshape(-1,1)).ravel()
    y_val_s   = scaler_y.transform(y_val.reshape(-1,1)).ravel()

    joblib.dump(scaler_X, os.path.join(MODEL_DIR, 'scaler_X.pkl'))
    joblib.dump(scaler_y, os.path.join(MODEL_DIR, 'scaler_y.pkl'))
    # Simpan juga max time_index agar bisa diprediksi untuk tanggal masa depan
    joblib.dump({'min_date': df['TANGGAL'].min()}, os.path.join(MODEL_DIR,'time_meta.pkl'))
    print("    Scaler tersimpan.")
    return (X_train_s, y_train_s, X_val_s, y_val_s,
            X_test_s, y_test,
            X_train, y_train,
            scaler_X, scaler_y, df, df_test, df_train)


def safe_mape(y_true, y_pred, threshold=MAPE_THRESHOLD):
    mask = np.abs(y_true) >= threshold
    if mask.sum() == 0:
        return None
    return round(float(np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100), 4)


def compute_metrics(y_true, y_pred, name="", include_mape=False):
    y_pred = np.clip(y_pred, 0, None)
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    m = {"MAE": round(mae,4), "RMSE": round(rmse,4), "R2": round(r2,4)}
    if include_mape:
        mape = safe_mape(y_true, y_pred)
        if mape is not None:
            m["MAPE"] = mape
    msg = f"    {name} → MAE:{mae:.4f} RMSE:{rmse:.4f} R²:{r2:.4f}"
    if "MAPE" in m:
        msg += f" MAPE:{m['MAPE']:.2f}%"
    print(msg)
    return m


# ============================================================
# BAGIAN 2 — MODEL 1: LINEAR REGRESSION
# ============================================================

def train_linear_regression(X_train_raw, y_train, X_test_s, y_test, scaler_X):
    """Ridge + Polynomial(deg=2, interaction_only) dilatih di y asli."""
    print("\n[2/7] LINEAR REGRESSION (Ridge + Polynomial + Trend)...")
    pipe = Pipeline([
        ('poly',  PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ('ridge', Ridge(alpha=1.0))
    ])
    pipe.fit(X_train_raw, y_train)

    X_test_raw = scaler_X.inverse_transform(X_test_s)
    y_pred = pipe.predict(X_test_raw)
    metrics = compute_metrics(y_test, y_pred, "Linear Regression (Ridge+Poly+Trend)")

    joblib.dump(pipe, os.path.join(MODEL_DIR, 'linear_regression.pkl'))
    print("    Model tersimpan → models/linear_regression.pkl")
    return metrics


# ============================================================
# BAGIAN 3 — MODEL 2: ANN
# ============================================================

def train_ann(X_train, y_train_s, X_val, y_val_s, X_test, y_test, scaler_y):
    print("\n[3/7] ANN (Artificial Neural Network)...")
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    tf.random.set_seed(42)
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(), Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(), Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='huber')

    history = model.fit(
        X_train, y_train_s,
        validation_data=(X_val, y_val_s),
        epochs=300, batch_size=32, verbose=0,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)
        ]
    )
    print(f"    Selesai {len(history.history['loss'])} epoch")

    y_pred_s = model.predict(X_test, verbose=0).ravel()
    y_pred   = scaler_y.inverse_transform(y_pred_s.reshape(-1,1)).ravel()
    metrics  = compute_metrics(y_test, y_pred, "ANN")

    model.save(os.path.join(MODEL_DIR, 'ann_model.h5'))
    model.save(os.path.join(MODEL_DIR, 'ann_model.keras'))
    json.dump({'loss':[float(v) for v in history.history['loss']],
               'val_loss':[float(v) for v in history.history['val_loss']]},
              open(os.path.join(MODEL_DIR,'ann_history.json'),'w'))
    print("    Model tersimpan → models/ann_model.h5")
    return metrics


# ============================================================
# BAGIAN 4 — MODEL 3: RNN/LSTM
# ============================================================

def create_sequences(X, y, timesteps=7):
    Xs, ys = [], []
    for i in range(len(X)-timesteps):
        Xs.append(X[i:i+timesteps]); ys.append(y[i+timesteps])
    return np.array(Xs), np.array(ys)


def train_lstm(X_train, y_train_s, X_val, y_val_s, X_test, y_test, scaler_y):
    print("\n[4/7] RNN/LSTM...")
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    TIMESTEPS = 7
    tf.random.set_seed(42)

    X_tr_seq, y_tr_seq = create_sequences(X_train, y_train_s, TIMESTEPS)
    X_vl_seq, y_vl_seq = create_sequences(X_val, y_val_s, TIMESTEPS)
    y_test_s = scaler_y.transform(y_test.reshape(-1,1)).ravel()
    X_te_seq, _ = create_sequences(X_test, y_test_s, TIMESTEPS)
    y_true_seq   = y_test[TIMESTEPS:]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(TIMESTEPS, X_train.shape[1])),
        Dropout(0.2),
        LSTM(32), Dropout(0.2),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='huber')

    history = model.fit(
        X_tr_seq, y_tr_seq,
        validation_data=(X_vl_seq, y_vl_seq),
        epochs=300, batch_size=32, verbose=0,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)
        ]
    )
    print(f"    Selesai {len(history.history['loss'])} epoch")

    y_pred_s = model.predict(X_te_seq, verbose=0).ravel()
    y_pred   = scaler_y.inverse_transform(y_pred_s.reshape(-1,1)).ravel()
    metrics  = compute_metrics(y_true_seq, y_pred, "LSTM", include_mape=True)

    model.save(os.path.join(MODEL_DIR, 'rnn_lstm_model.h5'))
    model.save(os.path.join(MODEL_DIR, 'rnn_lstm_model.keras'))
    json.dump({'loss':[float(v) for v in history.history['loss']],
               'val_loss':[float(v) for v in history.history['val_loss']]},
              open(os.path.join(MODEL_DIR,'lstm_history.json'),'w'))
    json.dump({"timesteps": TIMESTEPS, "buffer": X_train[-TIMESTEPS:].tolist()},
              open(os.path.join(MODEL_DIR,'lstm_seq_buffer.json'),'w'))
    print("    Model tersimpan → models/rnn_lstm_model.h5")
    return metrics


# ============================================================
# BAGIAN 5 — MODEL 4: K-MEANS CLUSTERING
# ============================================================

def train_kmeans(df):
    print("\n[5/7] K-MEANS CLUSTERING...")
    cluster_features = ['FF_AVG','FF_X','RH_AVG','TAVG','RR']
    df_clust = df[cluster_features].dropna().copy()

    scaler_clust = StandardScaler()
    X_clust = scaler_clust.fit_transform(df_clust)
    joblib.dump(scaler_clust, os.path.join(MODEL_DIR,'scaler_kmeans.pkl'))

    inertias, sil_scores, k_range = [], [], range(2,11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_clust)
        inertias.append(float(km.inertia_))
        sil_scores.append(float(silhouette_score(X_clust, labels)))

    best_k = int(k_range[np.argmax(sil_scores)])
    print(f"    K optimal (Silhouette): {best_k}")

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km_final.fit_predict(X_clust)
    sil = silhouette_score(X_clust, labels)

    df_clust = df_clust.copy(); df_clust['cluster'] = labels
    cluster_means = df_clust.groupby('cluster')['FF_AVG'].mean().sort_values()
    all_lbl = ["Angin Tenang","Angin Ringan","Angin Sedang","Angin Kencang",
               "Angin Sangat Kencang","Angin Badai","Angin Badai Kuat",
               "Angin Ekstrem","Angin Sangat Ekstrem"]
    label_map = {int(cid): all_lbl[r] for r,(cid,_) in enumerate(cluster_means.items())}

    df_full = df[['TANGGAL']+cluster_features].dropna().copy()
    df_full['cluster'] = labels
    df_full['cluster_label'] = df_full['cluster'].map(label_map)
    df_full.to_csv(os.path.join(DATA_DIR,'clustered_data.csv'), index=False)
    joblib.dump(km_final, os.path.join(MODEL_DIR,'kmeans_model.pkl'))

    km_metrics = {"Inertia": round(float(km_final.inertia_),4),
                  "Silhouette": round(float(sil),4), "K": best_k,
                  "label_map": label_map,
                  "elbow": {"k":list(k_range),"inertia":inertias,"silhouette":sil_scores}}
    json.dump(km_metrics, open(os.path.join(MODEL_DIR,'kmeans_metrics.json'),'w'))
    print(f"    Silhouette:{sil:.4f} K={best_k} | {label_map}")
    print("    Model tersimpan → models/kmeans_model.pkl")
    return km_metrics


# ============================================================
# BAGIAN 6 — MODEL 5: BACKPROPAGATION MANUAL (NumPy)
# ============================================================

class BackpropNetwork:
    """NumPy backprop + Adam + L2, arsitektur ramping untuk dataset kecil."""
    def __init__(self, layer_sizes, lr=0.001, l2=1e-4, seed=42):
        np.random.seed(seed)
        self.lr=lr; self.l2=l2; self.layer_sizes=layer_sizes
        self.weights=[]; self.biases=[]
        self.mw=[]; self.vw=[]; self.mb=[]; self.vb=[]
        self.t=0; self.b1=0.9; self.b2=0.999; self.eps=1e-8
        for i in range(len(layer_sizes)-1):
            W=np.random.randn(layer_sizes[i],layer_sizes[i+1])*np.sqrt(2.0/layer_sizes[i])
            b=np.zeros((1,layer_sizes[i+1]))
            self.weights.append(W); self.biases.append(b)
            self.mw.append(np.zeros_like(W)); self.vw.append(np.zeros_like(W))
            self.mb.append(np.zeros_like(b)); self.vb.append(np.zeros_like(b))

    def relu(self,z):  return np.maximum(0,z)
    def drelu(self,z): return (z>0).astype(float)

    def forward(self,X):
        self.acts=[X]; self.zs=[]
        a=X
        for i,(W,b) in enumerate(zip(self.weights,self.biases)):
            z=a@W+b; self.zs.append(z)
            a=z if i==len(self.weights)-1 else self.relu(z)
            self.acts.append(a)
        return self.acts[-1]

    def backward(self,y_true,clip=5.0):
        m=y_true.shape[0]; self.t+=1
        delta=(self.acts[-1]-y_true.reshape(-1,1))/m
        for i in reversed(range(len(self.weights))):
            dW=np.clip(self.acts[i].T@delta+self.l2*self.weights[i],-clip,clip)
            db=np.clip(np.sum(delta,axis=0,keepdims=True),-clip,clip)
            self.mw[i]=self.b1*self.mw[i]+(1-self.b1)*dW
            self.vw[i]=self.b2*self.vw[i]+(1-self.b2)*dW**2
            self.mb[i]=self.b1*self.mb[i]+(1-self.b1)*db
            self.vb[i]=self.b2*self.vb[i]+(1-self.b2)*db**2
            mwh=self.mw[i]/(1-self.b1**self.t); vwh=self.vw[i]/(1-self.b2**self.t)
            mbh=self.mb[i]/(1-self.b1**self.t); vbh=self.vb[i]/(1-self.b2**self.t)
            self.weights[i]-=self.lr*mwh/(np.sqrt(vwh)+self.eps)
            self.biases[i] -=self.lr*mbh/(np.sqrt(vbh)+self.eps)
            if i>0: delta=(delta@self.weights[i].T)*self.drelu(self.zs[i-1])

    def train_epoch(self,X,y,bs=32):
        idx=np.random.permutation(len(X)); X,y=X[idx],y[idx]; losses=[]
        for s in range(0,len(X),bs):
            Xb,yb=X[s:s+bs],y[s:s+bs]
            losses.append(float(np.mean((self.forward(Xb).ravel()-yb)**2)))
            self.backward(yb)
        return float(np.mean(losses))

    def predict(self,X): return self.forward(X).ravel()

    def save(self,path):
        d={f'W{i}':w for i,w in enumerate(self.weights)}
        d.update({f'b{i}':b for i,b in enumerate(self.biases)}); np.savez(path,**d)

    @classmethod
    def load(cls,path,layer_sizes):
        net=cls(layer_sizes); data=np.load(path)
        net.weights=[data[f'W{i}'] for i in range(len(layer_sizes)-1)]
        net.biases =[data[f'b{i}'] for i in range(len(layer_sizes)-1)]
        return net


def train_backprop(X_train, y_train_s, X_val, y_val_s, X_test, y_test, scaler_y):
    print("\n[6/7] BACKPROPAGATION MANUAL (NumPy + Adam + L2)...")
    layer_sizes=[X_train.shape[1],64,32,1]
    net=BackpropNetwork(layer_sizes,lr=0.001,l2=1e-4)
    best_val=float('inf'); best_w=None; best_b=None; pc=0
    hl=[]; hv=[]
    for epoch in range(1000):
        tl=net.train_epoch(X_train,y_train_s,bs=32)
        vp=net.predict(X_val); vl=float(np.mean((vp-y_val_s)**2))
        hl.append(tl); hv.append(vl)
        if vl<best_val:
            best_val=vl; pc=0
            best_w=[w.copy() for w in net.weights]
            best_b=[b.copy() for b in net.biases]
        else:
            pc+=1
            if pc>=50: print(f"    Early stopping epoch {epoch+1}"); break
    net.weights=best_w; net.biases=best_b
    y_pred_s=net.predict(X_test)
    y_pred=scaler_y.inverse_transform(y_pred_s.reshape(-1,1)).ravel()
    metrics=compute_metrics(y_test,y_pred,"Backpropagation")
    net.save(os.path.join(MODEL_DIR,'backprop_weights.npz'))
    json.dump({'loss':hl,'val_loss':hv,'layer_sizes':layer_sizes},
              open(os.path.join(MODEL_DIR,'backprop_history.json'),'w'))
    print("    Model tersimpan → models/backprop_weights.npz")
    return metrics


# ============================================================
# MAIN
# ============================================================

def main():
    print("="*60)
    print("  WINDPRED v8 — Training Pipeline (Perbaikan)")
    print("  Prediksi Kecepatan Angin BMKG Jabar")
    print("="*60)
    t0=time.time()

    (X_train_s, y_train_s, X_val_s, y_val_s,
     X_test_s, y_test,
     X_train_raw, y_train_raw,
     scaler_X, scaler_y, df, df_test, df_train) = load_and_preprocess()

    m_lr  = train_linear_regression(X_train_raw, y_train_raw, X_test_s, y_test, scaler_X)
    m_ann = train_ann(X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test, scaler_y)
    m_lstm= train_lstm(X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test, scaler_y)
    m_km  = train_kmeans(df)
    m_bp  = train_backprop(X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test, scaler_y)

    print("\n[7/7] MENYIMPAN METRIK...")
    candidates={"linear_regression":m_lr.get("R2",-999),"ann":m_ann.get("R2",-999),
                "lstm":m_lstm.get("R2",-999),"backprop":m_bp.get("R2",-999)}
    best=max(candidates,key=candidates.get)

    all_m={"linear_regression":m_lr,"ann":m_ann,"lstm":m_lstm,
           "kmeans":{"Inertia":m_km["Inertia"],"Silhouette":m_km["Silhouette"],"K":m_km["K"]},
           "backprop":m_bp,"best_model":best,"trained_at":datetime.now().isoformat()}
    json.dump(all_m,open(os.path.join(MODEL_DIR,'model_metrics.json'),'w'),indent=2)

    print("\n"+"="*60)
    print("  RINGKASAN")
    print("="*60)
    print(f"  Linear Regression → R²:{m_lr['R2']:.4f}  MAE:{m_lr['MAE']:.4f}")
    print(f"  ANN               → R²:{m_ann['R2']:.4f}  MAE:{m_ann['MAE']:.4f}")
    mape_s=f"  MAPE:{m_lstm.get('MAPE','—')}%" if m_lstm.get('MAPE') else ""
    print(f"  LSTM              → R²:{m_lstm['R2']:.4f}  MAE:{m_lstm['MAE']:.4f}{mape_s}")
    print(f"  K-Means           → Silhouette:{m_km['Silhouette']:.4f}  K={m_km['K']}")
    print(f"  Backpropagation   → R²:{m_bp['R2']:.4f}  MAE:{m_bp['MAE']:.4f}")
    print(f"\n  MODEL TERBAIK: {best.upper()}")
    print(f"  Total waktu: {time.time()-t0:.1f} detik")
    print("="*60)
    print("\nTraining selesai! Jalankan: python -m flask --app app/app.py run")

if __name__ == '__main__':
    main()
