# file_handler for ollama version

from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaLLM, OllamaEmbeddings
import os
import json

from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

PDF_STORAGE = "data/"
model = "qwen2"
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
embeddings = OllamaEmbeddings(model=model)
llm = OllamaLLM(model=model)

def check_file_exist(dir, file_name):
    file_name = file_name
    file_list = os.listdir(dir)
    file_exist = False
    for file in file_list:
        if file == file_name:
            file_exist = True
            return file_exist

def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE + uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def load_pdf(pdf_file):
    loader = PDFPlumberLoader(pdf_file)
    return loader.load()
   

def parse_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    page_count = len(reader.pages)
    # extract text from each page
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    split_text = text_splitter.split_text(text)
    return page_count, split_text

def split_into_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(docs)

def extract_text(docs):
    text = ""
    for doc in docs:
        text += doc.page_content
    return text

from langchain.prompts import PromptTemplate
# 요약문을 작성하기 위한 프롬프트 정의 (직접 프롬프트를 작성하는 경우)

def generate_question(docs):
    text = extract_text(docs)
    prompt = f"Please generate questions from the given {text}/,the questionnare should be in same language as given text"
  
    llm = ChatOpenAI(model= "gpt-4o-mini", temperature = 0.2)
    questions = llm.invoke(prompt)
    query_list = questions.content.split("\n")
    return query_list

def create_query_file(file_name, query_list):
    file_name = file_name.split('.')[0]
    query_file = "query/" + f"{file_name}" + "_query.txt"
    with open(query_file, "w", encoding="utf-8") as f:
        for query in query_list:
            f.write(query + "\n")
    return query_file

def create_qa_file(file_name, qa_pair):
    print(file_name)
    file_name = file_name.split('_query')[0].split('/')[1]
    print(file_name)
    qa_file = f"{file_name}" + "_qa.txt"
    if qa_file not in os.listdir("./qafiles"):
        data_list =[]
        with open(f"qafiles/{qa_file}", "w", encoding="utf-8") as json_file:
            json.dump(data_list, json_file, indent=4, ensure_ascii=False)
    with open(f"qafiles/{qa_file}", "r", encoding="utf-8") as json_file:
        data_list = json.load(json_file)
        data_list.append(qa_pair)
    with open(f"qafiles/{qa_file}", "w", encoding="utf-8") as json_file:
        json.dump(data_list, json_file, indent=4, ensure_ascii=False)
    return qa_file

def add_qa_file(file_name, qa_pair):
    #print(file_name)
    file_name = file_name.split('_query')[0].split('/')[1]
    qa_file = "qafiles/" + f"{file_name}" + "_qa.txt"

    # Check if the file exists
    if not os.path.exists(qa_file):
        # If the file doesn't exist, create an empty list
        qa_list = []
        with open(qa_file, "w", encoding="utf-8") as f:
            json.dump(qa_list, f, ensure_ascii=False, indent=4)
    # Read the existing data
    with open(qa_file, "r", encoding="utf-8") as f:
        qa_list = json.load(f)
        # Append the new Q&A pair to the list
    qa_list.append(qa_pair)
    # Write the updated list back to the file
    with open(qa_file, "w", encoding="utf-8") as f:
        json.dump(qa_list, f, ensure_ascii=False, indent=4)

    return qa_file