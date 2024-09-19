# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 16:05:40 2024

@author: abc
"""

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import json
import os
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["QDRANT_API_KEY"] = os.getenv('QDRANT_API_KEY')
from bs4 import BeautifulSoup
import requests

# Function to extract tables from the page
def extract_tables(soup):
    """Extract all tables from the page."""
    tables = []
    for table in soup.find_all("table"):
        rows = []
        for row in table.find_all("tr"):
            columns = [col.get_text(strip=True) for col in row.find_all(["td", "th"])]
            rows.append(columns)
        tables.append(rows)
    return tables

# Function to extract images (including diagrams) from the page
def extract_images(soup, domain):
    """Extract all image URLs."""
    image_urls = []
    for img in soup.find_all("img"):
        img_url = img.get("src")
        if img_url:
            if img_url.startswith("/"):
                img_url = domain + img_url  # Convert relative to absolute URLs
            image_urls.append(img_url)
    return image_urls
def extract_hidden_fields(soup):
    """Extract hidden fields from the page."""
    hidden_fields = {}
    for input_tag in soup.find_all("input", type="hidden"):
        name = input_tag.get("name")
        value = input_tag.get("value")
        if name and value:
            try:
                # Try to parse the value as JSON
                parsed_value = json.loads(value.replace('&quot;', '"'))
                hidden_fields[name] = parsed_value
            except json.JSONDecodeError:
                hidden_fields[name] = value
    return hidden_fields
# Update your get_vector_store_from_url function to include structured data
def get_vector_store_from_url(url):
    # Load the webpage content
    loader = WebBaseLoader(url)
    document = loader.load()
    
    # Parse the webpage with BeautifulSoup for advanced extraction
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Extract structured data (tables and images)
    tables = extract_tables(soup)
    images = extract_images(soup, url)
    hidden_fields = extract_hidden_fields(soup)
    
    # Combine structured data with regular text
    structured_data = f"Tables: {tables}\nImages: {images}\nHidden Fields: {hidden_fields}\n"
    document[0].page_content += structured_data  # Append structured data to the document

    # Text splitting and embedding
    text_splitter  = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    
    # Create a vector store from chunks
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    qdrant_url = "https://f6c816ad-c10a-4487-9692-88d5ee23882a.europe-west3-0.gcp.cloud.qdrant.io:6333"
    
    client = QdrantClient(":memory:",timeout=30)
    client.create_collection(
        collection_name="WebsiteQA - RAG",
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
    )
    
    vector_store = QdrantVectorStore.from_documents(
        document_chunks,
        embedding,
        url = qdrant_url,
        api_key = os.getenv("QDRANT_API_KEY"),
        collection_name="WebsiteQA - RAG"
    )
    
    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI(
        model = "gpt-4o-mini",
        )
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(
            variable_name = "chat_history"),
        ("user","{input}"),
        ("user","Given the above conversation, generate a search query to look up in order to get information relevant to the conversation.")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_chain(retriever_chain):
    llm = ChatOpenAI(
        model = "gpt-4o-mini",
        )
    prompt = ChatPromptTemplate.from_messages([
        ("system", """The user asked a question based on the following context extracted from the 
         website: {context}. Use this information as your primary source to generate a response. 
         Go through the content of the context and look for the answers. If you don't find relevant 
         information in the context, just say that Please ask relevant questions!, Don't try to make up an answer."""),

    MessagesPlaceholder(variable_name="chat_history"),
        ("user","{input}")
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)
    

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
   
   
    #       Concatenate the answer with the context
    full_response = response['answer']
   
    return full_response

    
st.set_page_config(page_title="Chat with Websites")
st.title("Website QA")
if "chat_history" not in st.session_state:
    st.session_state.chat_history  = [
        AIMessage(content="Hello, I am a bot. How can I help you?")
    ]

with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Enter Website URL")

    
if website_url is None or website_url == "":
    st.info("Please enter a website url")
else:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history  = [
            AIMessage(content="Hello, I am a bot. How can I help you?")
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vector_store_from_url(website_url)
        
    # user input
    user_query = st.chat_input("Type your question here..")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        
        
    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)