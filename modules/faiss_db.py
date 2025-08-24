# modules/vectorstore.py
import numpy as np
import pickle
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import faiss

# Initialize FAISS index (change dim to match your embedding size)

#index = faiss.IndexFlatL2(dim)
#embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
ollama_embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)

# 임베딩 차원 크기를 계산
dimension_size = len(ollama_embeddings.embed_query("hello world"))
print(dimension_size)
# Load existing index and metadata if available

# FAISS 벡터 저장소 생성
db = FAISS(
    embedding_function=ollama_embeddings,
    index=faiss.IndexFlatL2(dimension_size),
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# add documents to faiss vector store
def store_documents(documents):
    result = db.add_documents(documents = documents
    )
    return result
         

def search_similarity(query, k=1):
    results = db.similarity_search(query, k=k) 
    return results
