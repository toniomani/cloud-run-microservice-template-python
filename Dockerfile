FROM python:3.10-slim

WORKDIR /app

# --- NUEVO PASO ---
# Instalar los compiladores de C y herramientas de sistema para poder instalar River
RUN apt-get update && apt-get install -y build-essential gcc g++

# Copiar los archivos del proyecto
COPY . .

# Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto que usa Cloud Run
EXPOSE 8080

# Comando para arrancar Streamlit en el puerto de Cloud Run
CMD ["sh", "-c", "streamlit run app.py --server.port 8080 --server.address 0.0.0.0"]
