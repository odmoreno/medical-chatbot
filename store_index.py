"""
Script para indexar documentos médicos en Pinecone Vector Store
Este script procesa archivos PDF, los convierte en embeddings y los almacena en Pinecone
para su uso posterior en un chatbot médico con búsqueda semántica.
"""

# ============================================================================
# PASO 1: Importar dependencias necesarias
# ============================================================================
from dotenv import load_dotenv
import os
from src.helpers import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec 
from langchain_pinecone import PineconeVectorStore

# ============================================================================
# PASO 2: Cargar variables de entorno desde archivo .env
# ============================================================================
load_dotenv()

# ============================================================================
# PASO 3: Obtener y configurar las API keys necesarias
# ============================================================================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configurar las variables de entorno para que las librerías las puedan usar
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ============================================================================
# PASO 4: Cargar y procesar documentos PDF
# ============================================================================
# 4.1: Extraer texto de todos los archivos PDF en el directorio 'data/'
extracted_data = load_pdf_file(data='data/')

# 4.2: Filtrar documentos para mantener solo el contenido mínimo necesario
filter_data = filter_to_minimal_docs(extracted_data)

# 4.3: Dividir los documentos en chunks de texto más pequeños para procesamiento
text_chunks = text_split(filter_data)

# ============================================================================
# PASO 5: Descargar e inicializar el modelo de embeddings de Hugging Face
# ============================================================================
# Este modelo convierte texto en vectores numéricos (embeddings) para búsqueda semántica
embeddings = download_hugging_face_embeddings()

# ============================================================================
# PASO 6: Inicializar la conexión con Pinecone
# ============================================================================
pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

# ============================================================================
# PASO 7: Crear o verificar la existencia del índice en Pinecone
# ============================================================================
index_name = "medical-chatbot"  # Nombre del índice (cambiar si se desea)

# Verificar si el índice ya existe, si no, crearlo
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,  # Dimensión de los vectores (debe coincidir con el modelo de embeddings)
        metric="cosine",  # Métrica de similitud: distancia coseno
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),  # Configuración del servidor
    )

# Obtener referencia al índice creado o existente
index = pc.Index(index_name)

# ============================================================================
# PASO 8: Almacenar los documentos procesados en Pinecone Vector Store
# ============================================================================
# Convierte los chunks de texto en embeddings y los almacena en el índice de Pinecone
# Esto permite realizar búsquedas semánticas sobre los documentos médicos
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings, 
)