# modules/pdf_parser.py

from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
import pytesseract
from PIL import Image
import io
import os
import json
model = "gpt-4o-mini"
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

#embeddings = OpenAIEmbeddings()
ollama_embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    # model="chatfire/bge-m3:q8_0" # BGE-M3
)

def load_pdf(pdf_file):
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()
    #split_docs = text_splitter.split_documents(documents)
    # split texts will be embedded in next step
    return documents

def load_pdf_with_page_number(pdf_file):
    reader = PdfReader(pdf_file)
    documents = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        # Get the last non-empty line (likely the page number)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        page_number = lines[-1] if lines else str(i + 1)
        # Remove the page number from the text if needed
        if lines and lines[-1] == page_number:
            text = '\n'.join(lines[:-1])
        # Create a Document with metadata
        doc = Document(page_content=text, metadata={"page_number": page_number})
        documents.append(doc)
    return documents

def load_image_with_page_number(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    # Get the page number from the image file name (assuming format: foldername/filename_pageX.jpg)
    page_number = image_file.split("_")[-1].split(".")[0]
    doc = Document(page_content=text, metadata={"page_number": page_number})
    return [doc]


def parse_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    page_count = len(reader.pages)
    # extract text from each page
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    #split_text = text_splitter.split_text(text)
    return page_count, text

   
def save_list_to_file(file_name, data_list):
    with open(file_name, "w", encoding="utf-8") as file:
        file.write("\n".join(data_list))  # Join the list with newline and write to the file
    print(f"File '{file_name}' created successfully!")
    
from langchain.prompts import PromptTemplate
# 요약문을 작성하기 위한 프롬프트 정의 (직접 프롬프트를 작성하는 경우)

def generate_question(text):
    prompt = f"Please generate questions from the given {text}/, the given texts are mostly for clinical lab test for human illness,/ all questions should be in list and same language as given text"
  
    llm = ChatOpenAI(model= model, temperature = 0.2)
    questions = llm.invoke(prompt)
    query_list = questions.content.split("\n")
    return query_list

def create_query_file(file_name, query_list):
    file_name = file_name.split('.')[0]
    query_file = "query/" + f"{file_name}" + "_query.txt"
    
    # Writing the query list to a JSON file
    with open(query_file, "w", encoding="utf-8") as f:
        json.dump(query_list, f, ensure_ascii=False, indent=4)  # Dump the list as a JSON array
    return query_file

def create_qa_file(file_name, qa_pair):
    #print(file_name)
    file_name = file_name.split('_query')[0].split('/')[1]
    #print(file_name)
    qa_file = f"{file_name}" + "_qapair.txt"
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
    
def generate_question_ollama(text):
    llm = ChatOllama(model="llama3.2:latest", temperature = 0.2)
    prompt1 = f"Please generate questions from the given {text}/,the questionnare should be in same language as given text"
    prompt2 = f"You are a helpful questionnare creator. Please generate 10 questionnares that can be used for query from the context of given{text}/n, the questionnare should be in korean language as much possible, but the medical and chemical terminologies should remain in english"
    prompt = ChatPromptTemplate.from_template(
    "Please generate questions from the given {text}/, the questionnaire should be in same language as given text"
    )
    chain = prompt | llm | StrOutputParser()
    questions = chain.invoke({"text": text})
    #print(questions.content)
    return questions

def extract_text_from_image(image_path):
    # Use OCR to extract text from the image
    text = pytesseract.image_to_string(Image.open(image_path))
    return text