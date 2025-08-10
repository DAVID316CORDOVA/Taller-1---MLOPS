# ============================
# Imagen base ligera de Python
# ============================
# Se usa la versión slim para reducir el tamaño de la imagen.
FROM python:3.11-slim

# ============================
# Directorio de trabajo
# ============================
# Se define la carpeta donde residirá el código y se ejecutarán los comandos.
WORKDIR /app

# ============================
# Instalación de dependencias del sistema
# ============================
# Se instalan únicamente las herramientas mínimas necesarias (ej: curl).
# Luego se limpia la caché para reducir el tamaño final de la imagen.
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ============================
# Copia de dependencias Python
# ============================
# Se copia primero el requirements.txt para aprovechar la caché de Docker
# y acelerar compilaciones cuando no cambian las dependencias.
COPY requirements.txt .

# ============================
# Instalación de dependencias Python
# ============================
# Se actualiza pip y se instalan las dependencias indicadas.
# --no-cache-dir evita almacenar archivos temporales.
# --prefer-binary acelera la instalación usando binarios precompilados.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefer-binary -r requirements.txt

# ============================
# Copia de archivos de la aplicación
# ============================
# Se copian el código fuente y los modelos entrenados.
COPY app/ ./app/
COPY models/ ./models/

# ============================
# Verificación de contenido
# ============================
# Se listan los directorios para confirmar que los archivos fueron copiados correctamente.
RUN echo "Contenido de /app:" && ls -la /app && \
    echo "Contenido de /app/models:" && ls -la /app/models && \
    echo "Contenido de /app/app:" && ls -la /app/app

# ============================
# Seguridad: usuario no root
# ============================
# Se crea un usuario con UID 1000 y se le asigna la propiedad de /app.
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# ============================
# Exponer puerto
# ============================
# Se expone el puerto donde la aplicación escuchará peticiones HTTP.
EXPOSE 8000

# ============================
# Variables de entorno
# ============================
# PYTHONPATH para incluir /app en la ruta de módulos.
# PYTHONUNBUFFERED=1 para que los logs se muestren en tiempo real.
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# ============================
# Comando de ejecución
# ============================
# Se usa uvicorn para levantar la aplicación FastAPI en modo producción.
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

