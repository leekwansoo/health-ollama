import streamlit as st 
import io
import os
import json 
from modules.pdf_reader import generate_question_ollama, generate_question, parse_pdf, create_query_file, load_pdf, load_pdf_with_page_number, create_qa_file
from modules.pdf_reader import extract_text_from_image, load_image_with_page_number
from modules.faiss_db import store_documents # type: ignore
from modules.query_handler import query_library, query_chain, check_query_exist
from doc_handler import add_document, retrieve_document, list_documents, check_document
from graph import search_web
from dotenv import load_dotenv
load_dotenv()
from langchain_core.documents import Document
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
st.title("Webzine for SCL Health")

# Main Page content 
def main_query(file_name=None, default_query=""):
    document_name = file_name
    query = st.text_input("Enter your question for your uploaded documents:", value=default_query)
    # Pronounce button using gTTS (Google Text-to-Speech)
    from io import BytesIO
    try:
        from gtts import gTTS
        tts_available = True
    except ImportError:
        tts_available = False

    if tts_available and st.button("🔊 Pronounce Query", key="pronounce_query"):
        if query:
            tts = gTTS(text=query, lang='en')
            mp3_fp = BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            st.audio(mp3_fp, format='audio/mp3')
    elif not tts_available:
        st.info("Install gTTS to enable pronunciation: pip install gTTS")

    if st.button("Submit", key="submit_query"):
        if query:
            if document_name:
                # check if the query is exiting in the qafiles folder with document_name and query content
                result = check_query_exist(query, document_name=document_name)
                if result:
                    st.write(result)
                    st.write("this answer is taken from previous query")
                    st.markdown(f"**Referenced source:** `{document_name}`")
                    st.download_button("Download Response", data=json.dumps(result, ensure_ascii=False, indent=4), file_name="response.txt", mime="application/json")
                else:
                    qa_pair = query_chain(query, document_name=document_name)
                    if qa_pair:
                        st.write(qa_pair)
                    # Display the referenced source at the end
                    if document_name:
                        st.markdown(f"**Referenced source:** `{document_name}`")

                        st.download_button("Download Response", data=json.dumps(qa_pair, ensure_ascii=False, indent=4), file_name="response.txt", mime="application/json")
            else:
                st.write("No document uploaded.")

        #response = query_library(query)
        #if response:
        #    qa_pair = {"query": query, "answer": response.content}
        #    qa_file = create_qa_file(file_name, qa_pair)
        #    st.write(response.content)
        #    st.write(f"QA pair is saved in {qa_file}")
            
        
st.session_state["query_message"] = []
st.session_state["query_file"] = []

# Create a sidebar for navigation
st.sidebar.title("Menu")
options = st.sidebar.radio("Select an option", ["Upload File", "Upload_Image Folder", "Query from Uploaded File", "Web Search"])

if options == "Upload File":
    st.sidebar.header("Upload File")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf", key="pdf_uploader")
       
    if uploaded_file:
        file_name = uploaded_file.name
        check_exist = check_document(file_name)
        #check_exist = "noexist"
        if check_exist == "noexist":
            # store the file in the uploaded file folder
            uploaded_name = f"uploaded/{file_name}"
            with open(uploaded_name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            #page_count, text = parse_pdf(uploaded_file)

            documents = load_pdf_with_page_number(uploaded_name)
            #print(documents)
            result = store_documents(documents)
            if result:
                st.sidebar.write("Data is embedded in the Faiss_db")
                add_document(file_name)
                query_lists = []
                for doc in documents:
                    query_list = generate_question_ollama(doc.page_content)
                    print(query_list)
                    query_lists.append(query_list)
                query_file = create_query_file(file_name, query_lists)
                st.session_state["query_file"].append(query_file)
                #for query in query_list:
                st.session_state["query_message"].append(query_lists)
                #st.markdown(query_lists)
                st.markdown('\n\n'.join(query_lists))

            else:
                st.sidebar.write("storing PDF file into vector store failed")
                      
        else:
            st.sidebar.write(f"{file_name} is aleardy uploaded\n")     
         
    else:
        st.sidebar.write("Please upload a PDF and select subject to get started.")
        
    main_query()
    
elif options == "Upload_Image Folder":
    st.sidebar.header("Upload Image Folder")
    uploaded_files = st.sidebar.file_uploader(
        "Upload multiple JPG images (select all images in a folder)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="folder_uploader"
    )
    if uploaded_files:
        # Try to infer the folder name from the uploaded file names (assuming format: foldername/filename.jpg)
        # If not possible, fallback to 'images_folder'
        first_file = uploaded_files[0].name if uploaded_files else None
        print("First File:", first_file)
        if first_file:
            directory_name = first_file.split("_")[0]
            print("Directory_Name:", directory_name)
        else:
            directory_name = "images_folder"
        folder_path = f"uploaded/{directory_name}"
        os.makedirs(folder_path, exist_ok=True)
        all_text = ""
        documents = []
        for file in uploaded_files:
            # if file is in the folder_path already, skip the processing
            if os.path.exists(os.path.join(folder_path, file.name)):
                st.sidebar.write(f"File {file.name} already exists. Skipping...")
            else:
            # If file.name contains a subdirectory, use only the filename for saving
                file_save_name = file.name.split("/")[-1]
                file_path = os.path.join(folder_path, file_save_name)
                print("Saved File Path:", file_path)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                docs = load_image_with_page_number(file_path)
                all_text += "\n".join([doc.page_content for doc in docs]) + "\n"
                documents.extend(docs)
                result = store_documents(documents)
                if result:
                    st.sidebar.write(f"All images in folder are embedded in the Faiss_db")
                    query_list = generate_question(all_text)
                    print(query_list)
                    query_file = create_query_file(directory_name + ".txt", query_list)
                    st.session_state["query_file"].append(query_file)
                    
                    st.session_state["query_message"].append(query_list)
                    st.markdown('\n\n'.join(query_list))
                else:
                    st.sidebar.write("Storing images into vector store failed")
    else:
        st.sidebar.write("Please upload multiple JPG images from a folder.")
    main_query()
    
elif options == "Query from Uploaded File":
    st.header("Query from Uploaded File")
    query_list = os.listdir("query")
    selected = st.sidebar.selectbox("Select document to query", query_list)
    file_name = f"query/{selected}"
    with open(file_name, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]

    if "selected_question" not in st.session_state:
        st.session_state["selected_question"] = ""

    import re
    for idx, question in enumerate(questions):
        # Remove leading numbering, quotes, and whitespace, and trailing quotes/commas
        display_question = re.sub(r'^\s*\d+\.\s*', '', question)
        # Remove trailing comma and/or quote, regardless of order
        display_question = re.sub(r'[",]+$', '', display_question).strip()
        col1, col2 = st.sidebar.columns([8, 2])
        col1.markdown(display_question)
        if display_question.strip().endswith("?"):
            if col2.button("Query", key=f"query_btn_{idx}"):
                st.session_state["selected_question"] = display_question

    main_query(file_name=file_name, default_query=st.session_state["selected_question"])
          
elif options == "Web Search":
    st.header("Web Search")
    query = st.text_input("Enter a search query:")
    if st.button("Search Web"):
        if query:
            results = search_web(query)
            for result in results:
                st.write(result["content"])
        else:
            st.write("Please enter a search query.")