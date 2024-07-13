# Dockerfile
FROM python:slim

# Crear y establecer directorio de trabajo
WORKDIR /app

# Copiar y instalar dependencias
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código de la aplicación
COPY . /app

# Exponer el puerto que Streamlit utilizará
EXPOSE 8501

# Configurar el comando de inicio de la aplicación
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]