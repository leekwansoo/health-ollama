import streamlit as st
import os
st.session_state["DOCUMENT"] =[]
st.session_state["DOCUMENT"] = os.listdir("uploaded")
print(st.session_state["DOCUMENT"])

st.session_state["DOCUMENT"] = []
st.session_state["DOCUMENT"] = os.listdir("uploaded")
#print(st.session_state["DOCUMENT"])
doc_list =[]
#if st.session_state["DOCUMENT"] is None:
#   st.session_state["DOCUMENT"] = {"key": ""}
    
# Function to add data to DOCUMENT directory 
def check_document(value): 
    if value not in st.session_state["DOCUMENT"]:
        result = "noexist"
        return(result)
    else:
        result = "exist"
        return(result)
       
def add_document(value):  
    st.session_state["DOCUMENT"] = os.listdir("uploaded")       
    st.session_state["DOCUMENT"].append(value)
    st.write(f"Document added: {value}") 
# Function to retrieve data from DOCUMENT directory 
def retrieve_document(doc): 
    if doc in st.session_state["DOCUMENT"]: 
        return doc
    else: st.write(f"Document with key '{doc}' not found.") 
    return None 
# Example usage # List Document
def list_documents(): 
    if st.session_state["DOCUMENT"]:
        docs = st.session_state["DOCUMENT"]
        print(docs)
        return(docs)
    else: st.write("No documents found in 'DOCUMENT'.")

"""
# Adding data 
add_document("doc1", "This is the content of document 1.")\n 
add_document("doc2", "This is the content of document 2.") 

# Retrieving data 
retrieve_document("doc1")\n 
"""
