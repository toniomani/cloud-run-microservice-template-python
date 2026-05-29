import streamlit as st
import pandas as pd
import numpy as np
import io
import pickle
from google.cloud import storage
from river import linear_model, preprocessing, metrics, optim

# =========================================================
# CONFIGURACIÓN 
# =========================================================
st.set_page_config(page_title="Aprendizaje en línea", page_icon="🚕")
st.title("Aprendizaje en línea con River (Step-by-step desde GCS)")

st.markdown("""
Este panel permite entrenar un modelo de **aprendizaje incremental** con River,
procesando **un archivo por clic** desde Google Cloud Storage (GCS).
""")

# =========================================================
# PARÁMETROS Y RUTAS GCS
# =========================================================
bucket_name = st.text_input("Bucket de GCS:", "bucket_131025_act3")
prefix = st.text_input("Prefijo/carpeta:", "tlc_yellow_trips_2022/")
limite = st.number_input("Filas a procesar por archivo:", value=1000, step=100)

MODEL_PATH = "models/model_incremental.pkl"
STATE_PATH = "models/state_incremental.pkl"  # <-- Archivo para persistir metadata

# =========================================================
# FUNCIONES AUXILIARES GCS
# =========================================================
def save_to_gcs(obj, bucket_name, destination_blob):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob)
        blob.upload_from_string(pickle.dumps(obj))
    except Exception as e:
        st.warning(f"No se pudo guardar en GCS `{destination_blob}`: {e}")

def load_from_gcs(bucket_name, source_blob):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob)
        if blob.exists():
            return pickle.loads(blob.download_as_bytes())
        return None
    except Exception as e:
        st.warning(f"No se pudo cargar desde GCS `{source_blob}`: {e}")
        return None

def delete_from_gcs(bucket_name, source_blob):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob)
        if blob.exists():
            blob.delete()
    except Exception as e:
        pass

# =========================================================
# MODELO BASE
# =========================================================
def new_model():
    return preprocessing.StandardScaler() | linear_model.LinearRegression(
        optimizer=optim.SGD(0.001),
        intercept_lr=0.001
    )

# =========================================================
# INICIALIZAR / RESTAURAR ESTADO COMPLETO DESDE GCS
# =========================================================
if "model" not in st.session_state:
    loaded_model = load_from_gcs(bucket_name, MODEL_PATH)
    loaded_state = load_from_gcs(bucket_name, STATE_PATH)

    if loaded_model is None or loaded_state is None:
        # Inicialización limpia (Primera vez o tras un reinicio)
        st.session_state.model = new_model()
        st.session_state.index = 0
        st.session_state.blobs = None
        st.session_state.metric_r2 = metrics.R2()
        st.session_state.metric_mae = metrics.MAE()
        st.session_state.history_r2 = []
        st.session_state.history_mae = []
        st.session_state.history_file_r2 = []
        st.session_state.history_file_mae = []
        st.session_state.processed_files = []
    else:
        # ¡RESTAURACIÓN ABSOLUTA DESDE GCS! Rompe la volatilidad de Cloud Run
        st.session_state.model = loaded_model
        st.session_state.index = loaded_state["index"]
        st.session_state.blobs = loaded_state["blobs"]
        st.session_state.metric_r2 = loaded_state["metric_r2"]
        st.session_state.metric_mae = loaded_state["metric_mae"]
        st.session_state.history_r2 = loaded_state["history_r2"]
        st.session_state.history_mae = loaded_state["history_mae"]
        st.session_state.history_file_r2 = loaded_state["history_file_r2"]
        st.session_state.history_file_mae = loaded_state["history_file_mae"]
        st.session_state.processed_files = loaded_state["processed_files"]

# Mapeo local rápido para legibilidad
model = st.session_state.model
metric_r2 = st.session_state.metric_r2
metric_mae = st.session_state.metric_mae

# =========================================================
# BOTÓN PARA REINICIAR TODO
# =========================================================
st.markdown("---")
if st.button("Reiniciar entrenamiento y borrar todo en GCS"):
    delete_from_gcs(bucket_name, MODEL_PATH)
    delete_from_gcs(bucket_name, STATE_PATH)
    st.session_state.clear()  # Vacía el session_state actual
    st.rerun()

# =========================================================
# FEATURE ENGINEERING Y PROCESAMIENTO
# =========================================================
def _parse_time_fields(row):
    for c in ("tpep_pickup_datetime", "lpep_pickup_datetime", "pickup_datetime"):
        if c in row and pd.notna(row[c]):
            dt = pd.to_datetime(row[c], errors="coerce")
            if pd.notna(dt): return dt, int(dt.hour)
    return None, 0

def _extract_x(row):
    dist = float(row["trip_distance"])
    dt, hour = _parse_time_fields(row)
    dow = int(dt.weekday()) if isinstance(dt, pd.Timestamp) else 0
    return {
        "dist": dist,
        "log_dist": float(np.log1p(max(dist, 0))),
        "pass": float(row["passenger_count"]),
        "hour": float(hour),
        "dow": float(dow),
        "is_weekend": 1.0 if dow >= 5 else 0.0,
    }

def process_single_blob(bucket_name, blob_name, limite=1000):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        buffer = io.BytesIO(blob.download_as_bytes())
        
        chunks = []
        for chunk in pd.read_csv(buffer, chunksize=500, low_memory=False):
            cols = ["trip_distance", "passenger_count", "fare_amount"]
            if not set(cols).issubset(chunk.columns): continue
            for c in cols: chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
            chunk = chunk.dropna(subset=cols)
            chunk = chunk[(chunk["fare_amount"].between(2, 200)) & (chunk["trip_distance"].between(0.1, 50))]
            if not chunk.empty: chunks.append(chunk)
            
        if not chunks: return None
        df_file = pd.concat(chunks, ignore_index=True)
        if len(df_file) > limite: df_file = df_file.sample(n=limite, random_state=42)
        
        file_r2, file_mae = metrics.R2(), metrics.MAE()
        count = 0
        
        for _, row in df_file.iterrows():
            y = float(row["fare_amount"])
            x = _extract_x(row)
            pred = model.predict_one(x)
            pred_eval = float(np.clip(pred, 2, 200)) if pred is not None else 0.0
            
            metric_r2.update(y, pred_eval)
            metric_mae.update(y, pred_eval)
            file_r2.update(y, pred_eval)
            file_mae.update(y, pred_eval)
            model.learn_one(x, y)
            count += 1
            
        return {"count": count, "file_r2": file_r2.get(), "file_mae": file_mae.get(), "global_r2": metric_r2.get(), "global_mae": metric_mae.get()}
    except Exception as e:
        st.warning(f"Error procesando archivo: {e}")
        return None

# =========================================================
# INTERFAZ DE PROCESAMIENTO INCREMENTAL
# =========================================================
st.subheader("Procesamiento incremental")

if st.button("Procesar siguiente archivo"):
    # Si la lista de blobs no existe en la sesión, la recuperamos
    if st.session_state.blobs is None:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs_list = list(bucket.list_blobs(prefix=prefix))
        # Guardamos strings, no objetos binarios complejos
        st.session_state.blobs = [b.name for b in blobs_list if b.name.endswith(".csv") and not b.name.endswith("/")]
        st.session_state.index = 0

    blobs = st.session_state.blobs
    idx = st.session_state.index

    if idx >= len(blobs):
        st.success("Todos los archivos ya fueron procesados.")
    else:
        blob_name = blobs[idx]
        short = blob_name.split("/")[-1]
        st.write(f"Procesando archivo {idx + 1}/{len(blobs)}: `{short}`")

        result = process_single_blob(bucket_name, blob_name, int(limite))

        if result is not None:
            st.session_state.history_r2.append(result["global_r2"])
            st.session_state.history_mae.append(result["global_mae"])
            st.session_state.history_file_r2.append(result["file_r2"])
            st.session_state.history_file_mae.append(result["file_mae"])
            st.session_state.processed_files.append(short)
            st.write(f"Registros procesados: **{result['count']}**")
        else:
            st.error(f"El archivo `{short}` no devolvió registros válidos.")

        # Avanzamos el índice obligatoriamente
        st.session_state.index += 1

        # Sincronización crucial: Guardamos modelo y todo el estado tracking en GCS
        save_to_gcs(model, bucket_name, MODEL_PATH)
        
        current_state = {
            "index": st.session_state.index,
            "blobs": st.session_state.blobs,
            "metric_r2": st.session_state.metric_r2,
            "metric_mae": st.session_state.metric_mae,
            "history_r2": st.session_state.history_r2,
            "history_mae": st.session_state.history_mae,
            "history_file_r2": st.session_state.history_file_r2,
            "history_file_mae": st.session_state.history_file_mae,
            "processed_files": st.session_state.processed_files
        }
        save_to_gcs(current_state, bucket_name, STATE_PATH)
        st.rerun()

# =========================================================
# RENDERIZADO DE MÉTRICAS E HISTORIAL
# =========================================================
st.markdown("---")
st.subheader("Estado actual del modelo")
st.write(f"Siguiente índice a procesar: **{st.session_state.index}**")
st.write(f"R² acumulado actual: **{metric_r2.get():.4f}**")
st.write(f"MAE acumulado actual: **{metric_mae.get():.4f}**")

if st.session_state.history_r2:
    df_hist = pd.DataFrame({
        "archivo": st.session_state.processed_files,
        "R2_archivo": st.session_state.history_file_r2,
        "MAE_archivo": st.session_state.history_file_mae,
        "R2_acumulado": st.session_state.history_r2,
        "MAE_acumulado": st.session_state.history_mae,
    })
    st.subheader("Historial de procesamiento")
    st.dataframe(df_hist)
    st.subheader("Evolución de métricas acumuladas")
    st.line_chart(df_hist[["R2_acumulado", "MAE_acumulado"]])
