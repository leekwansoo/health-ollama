# config.py
from dotenv import load_dotenv
load_dotenv()
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTORSTORE_PATH = 'vectorstore.faiss' 
METADATA_PATH = 'metadata.pkl'