# modules/qa_module.py

from openai import OpenAI
from modules.faiss_db import search_similarity
from langchain_core.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_ollama import ChatOllama

from dotenv import load_dotenv
load_dotenv()
import os
import json
#os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

client=OpenAI()

#def answer_question(question, context):
    # Use OpenAI (original code)
    #chat_completion = client.chat.completions.create(
       # messages=[{"role": "user", "content": question}],
       # model="gpt-4o-mini",
    #)
    #return chat_completion.choices[0].message['content'].strip()

def answer_question_ollama(question, context):
    # Use ChatOllama
    prompt = f"Given the context: {context}\n\nQ: {question}\nA:"
    model = ChatOllama(model="gemma:7b-instruct", temperature=0.2)
    response = model.invoke(prompt)
    return response.content.strip() if hasattr(response, 'content') else str(response).strip()


def create_prompt(query, document_name=None):   
    # Search query
    if document_name:
        #results = search_similarity(query, k=1, document_name=document_name)
        results = search_similarity(query, k=1)
    else:
        results = search_similarity(query, k=1)
    context = ""
    for result in results:
        context += result.page_content
    prompt = f"Given the context: {context}\n\nQ: {query}\nA:"
    return prompt

def query_library(query):
    prompt = create_prompt(query)
    print(prompt) 
    # Generate response using OpenAI 
    chat_completion = client.chat.completions.create( messages=[ {"role": "user", "content": prompt} ], model="gpt-4o-mini", ) 
    
    return chat_completion.choices[0].message

def query_chain(query, document_name=None):
    prompt = create_prompt(query, document_name=document_name)
    model = ChatOllama(model = "gemma:7b-instruct", temperature = 0.2)
    prompt_callable = lambda _: prompt  # Create a callable for the prompt
    chain = ( prompt_callable | model | StrOutputParser())
    response = chain.invoke({})
    qa_pair = {"query": query, "response": response}

    # Save qa_pair to qafiles/document_qapair.txt as a list
    if document_name:
        import os, json
        os.makedirs("qafiles", exist_ok=True)
        # Remove path and extension for file base name
        base_name = os.path.splitext(os.path.basename(document_name))[0].split("_")[0]
        qa_file = os.path.join("qafiles", f"{base_name}_qapair.txt")
        # Load existing list or create new
        if os.path.exists(qa_file):
            with open(qa_file, "r", encoding="utf-8") as f:
                try:
                    qa_list = json.load(f)
                except Exception:
                    qa_list = []
        else:
            qa_list = []
        qa_list.append(qa_pair)
        with open(qa_file, "w", encoding="utf-8") as f:
            json.dump(qa_list, f, ensure_ascii=False, indent=4)

    return qa_pair

def check_query_exist(query, document_name=None):
    # Check if the query exists in the QA files
    if document_name:
        base_name = os.path.splitext(os.path.basename(document_name))[0].split("_")[0]
        qa_file = os.path.join("qafiles", f"{base_name}_qapair.txt")
    else:
        qa_file = "qafiles/default_qapair.txt"
        print(qa_file)
    if os.path.exists(qa_file):
        with open(qa_file, "r", encoding="utf-8") as f:
            try:
                qa_list = json.load(f)
                for qa_pair in qa_list:
                    if qa_pair["query"] == query:
                        return qa_pair
            except Exception:
                return None
    return None