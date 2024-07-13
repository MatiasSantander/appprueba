import streamlit as st
import login

login.generarLogin()
if 'usuario' in st.session_state:
    st.header('Chat:red[BOT]')


import streamlit as st
from dotenv import load_dotenv
import os
from openai import OpenAI
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Load environment variables
load_dotenv()

# OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Pinecone API key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize the OpenAI chat model
model = ChatOpenAI(openai_api_key=api_key, model="gpt-4o")

# Initialize the output parser
parser = StrOutputParser()

# Define the prompt template
template = """
Answer the question based on the context below. If you can't 
answer the question, reply "No tengo esa información".

Context: {context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# Streamlit UI
st.title("Pinecone Chat")

# Initialize Pinecone
database = Pinecone(api_key=PINECONE_API_KEY)

# List available indexes
indexes = database.list_indexes().names()
selected_index = st.selectbox("Seleccionar base de datos vectorial", indexes)

if selected_index:
    pinecone_index = database.Index(selected_index)
    st.write(f"Conectado a la base de datos: {selected_index}")

    # Define the question input
    question = st.text_input("Ingresar pregunta")

    if question:
        # Initialize PineconeVectorStore with embeddings
        embeddings = OpenAIEmbeddings()
        
        # Define documents and index_name
        documents = []  # Replace with your actual documents
        index_name = selected_index

        pinecone = PineconeVectorStore.from_documents(
            documents, embeddings, index_name=index_name
        )

        # Define the retrieval and processing chain
        chain = (
            {"context": pinecone.as_retriever(), "question": RunnablePassthrough()}  # Retrieve context from Pinecone
            | prompt  # Apply the prompt template
            | model  # Use the model to generate a response
            | parser  # Parse the model's response
        )

        # Invoke the chain with the question
        response = chain.invoke(question)
        # Display the response
        st.write("Respuesta:")
        st.write(response)