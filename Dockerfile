FROM python:3.10-slim

WORKDIR /app

# Copiamos el archivo de dependencias desde su carpeta original
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# CORRECCIÓN: Copiamos absolutamente TODO el código del proyecto a la carpeta /app
# (Tu archivo .dockerignore se encargará de que NO se suban cosas pesadas como la data local)
COPY . .

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Ajustamos la ruta para que corra el main.py que está dentro de la carpeta api
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
