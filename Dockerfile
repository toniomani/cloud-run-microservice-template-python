# Usamos la imagen completa que ya trae todas las herramientas científicas y de compilación
FROM python:3.10

WORKDIR /app

# Copiar los archivos del proyecto
COPY . .

# Actualizar pip y las herramientas de construcción para evitar errores con librerías pesadas
RUN pip install --upgrade pip setuptools wheel

# Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto que usa Cloud Run
EXPOSE 8080

# Comando para arrancar Streamlit en el puerto de Cloud Run
CMD ["sh", "-c", "streamlit run app.py --server.port 8080 --server.address 0.0.0.0"]
