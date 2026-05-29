FROM python:3.10-slim

WORKDIR /app

# Copiar los archivos del proyecto
COPY . .

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto que usa Cloud Run
EXPOSE 8080

# Comando para arrancar Streamlit en el puerto de Cloud Run
CMD ["sh", "-c", "streamlit run app.py --server.port 8080 --server.address 0.0.0.0"]
