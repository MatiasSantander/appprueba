import streamlit as st
import login

login.generarLogin()
if 'usuario' in st.session_state:
    st.header('Descargar y :blue[procesar]')

import streamlit as st
from google.cloud import storage
from google.oauth2 import service_account
from pytubefix import YouTube
import tempfile
import os
import whisper
from dotenv import load_dotenv
import pandas as pd
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from openai import OpenAI
import re
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

# Google Cloud credentials
credentials_path = 'keyservice.json'
credentials = service_account.Credentials.from_service_account_file(credentials_path)
storage_client = storage.Client(credentials=credentials)

# Desactivar la barra de progreso de tqdm
os.environ["TQDM_DISABLE"] = "1"

# Whisper model
whisper_model = whisper.load_model("base")

# OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Pinecone API key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def upload_to_bucket(bucket_name, source_file_name, destination_blob_name):
    """Sube un archivo a un bucket de Google Cloud Storage."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    st.write(f"Archivo {source_file_name} subido a bucket {bucket_name}.")

def file_exists_in_bucket(bucket_name, blob_name):
    """Revisa si un archivo existe en un bucket de Google Cloud Storage."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.exists()

def get_embedding(sentence):
    return (
        client.embeddings.create(input=sentence, model="text-embedding-3-small")
        .data[0]
        .embedding
    )

# Función para limpiar el título del video
def clean_title(title):
    # Reemplazar caracteres no permitidos con guiones y convertir a minúsculas
    cleaned = re.sub(r'[^a-z0-9]+', '-', title.lower())
    # Asegurarse de que termine con un carácter alfanumérico
    if not cleaned[-1].isalnum():
        cleaned = cleaned.rstrip('-') + '0'
    return cleaned

# Streamlit UI
st.title("YouTube Audio Transcription")

# Input for YouTube URL
youtube_url = st.text_input("Ingresar URL de YouTube")

if youtube_url:
    youtube = YouTube(youtube_url)
    video_title = youtube.title
    cleaned_title = clean_title(video_title)[:45]  # Truncate to 45 characters
    audio = youtube.streams.filter(only_audio=True).first()
    audio_size = audio.filesize / (1024 * 1024)  # Convert to MB

    st.write(f"Titulo del video: {video_title}")
    st.write(f"Tamaño del archivo de audio: {audio_size:.2f} MB")

    # Input for Pinecone Index Name
    index_name_input = st.text_input("Ingresar nombre del índice de Pinecone")

    if index_name_input and st.button("Guardar audio en el bucket"):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_file = audio.download(output_path=tmpdir)
            bucket_name = 'ga4bucket123'
            destination_blob_name = f"{cleaned_title}.mp3"
            transcription_file = f"{cleaned_title}.txt"

            if not file_exists_in_bucket(bucket_name, destination_blob_name):
                upload_to_bucket(bucket_name, audio_file, destination_blob_name)
                st.write("Audio guardado en el bucket.")

                # Transcribe audio using Whisper
                transcription = whisper_model.transcribe(audio_file, fp16=False)["text"].strip()

                with open(transcription_file, "w") as file:
                    file.write(transcription)

                # Upload transcription to bucket
                upload_to_bucket(bucket_name, transcription_file, transcription_file)
                st.write("Transcripción guardada en el bucket.")
            else:
                st.write(f"El archivo {destination_blob_name} ya se encuentra en el bucket {bucket_name}.")
                # Descargar la transcripción existente
                blob = storage_client.bucket(bucket_name).blob(transcription_file)
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    blob.download_to_filename(temp_file.name)
                    with open(temp_file.name, "r") as file:
                        transcription = file.read()

            # Mostrar el elemento de audio y la transcripción
            st.audio(audio_file)
            st.text_area("Transcripción", transcription)

            # Cargar el documento de texto
            loader = TextLoader(transcription_file)
            text_documents = loader.load()

            # Dividir los documentos en fragmentos
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            documents = text_splitter.split_documents(text_documents)

            # Crear un DataFrame a partir de los documentos divididos
            data = {
                "id": [i for i in range(len(documents))],
                "text": [doc.page_content for doc in documents]
            }

            df = pd.DataFrame(data)

            # Aplicar la función get_embedding a la columna 'text' y crear una nueva columna 'embedding'
            df['embedding'] = df['text'].apply(get_embedding)

            # Mostrar las primeras filas del DataFrame
            #st.write(df.head())

            # Obtener la dimensión del embedding
            embedding_dimension = len(df.iloc[0]["embedding"])

            # Inicializar la base de datos de Pinecone
            database = Pinecone(api_key=PINECONE_API_KEY)

            # Especificaciones del servidor sin servidor
            serverless_spec = ServerlessSpec(cloud="aws", region="us-east-1")

            # Crear el índice en Pinecone
            INDEX_NAME = index_name_input

            # Asegurarse de que el nombre del índice termine con un carácter alfanumérico
            if not INDEX_NAME[-1].isalnum():
                INDEX_NAME += '0'

            # Eliminar un índice existente si se ha alcanzado el límite
            if len(database.list_indexes().names()) >= 5:
                index_to_delete = database.list_indexes().names()[0]  # Eliminar el primer índice (puedes cambiar la lógica)
                database.delete_index(index_to_delete)
                st.write(f"Índice {index_to_delete} eliminado para liberar espacio.")

            if INDEX_NAME not in database.list_indexes().names():
                database.create_index(
                    name=INDEX_NAME,
                    dimension=embedding_dimension,
                    metric="cosine",
                    spec=serverless_spec,
                )

                time.sleep(1)

            pinecone_index = database.Index(INDEX_NAME)

            # Definir el iterador y la función de vector
            def iterator(dataset, size):
                for i in range(0, len(dataset), size):
                    yield dataset.iloc[i : i + size]

            def vector(batch):
                vector = []
                for i in batch.to_dict("records"):
                    vector.append((str(i["id"]), i["embedding"], {"text": i["text"]}))

                return vector

            # Insertar los vectores en el índice de Pinecone
            if pinecone_index.describe_index_stats()["total_vector_count"] == 0:
                for batch in iterator(df, 100):
                    pinecone_index.upsert(vector(batch))

            # Describir las estadísticas del índice
            #st.write(pinecone_index.describe_index_stats())

            # Inicializar las embeddings de OpenAI
            #embeddings = OpenAIEmbeddings()

            # Configurar PineconeVectorStore
            #index_name = cleaned_title

            #pinecone = PineconeVectorStore.from_documents(
            #    documents, embeddings, index_name=index_name
            #)

            st.write("Base de datos activada y actualizada.")