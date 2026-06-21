from langchain_community.document_loaders import PyPDFLoader
import os

def load_all_pdfs():
    folder_path = "data/pdfs"
    num_docs = 0
    all_docs = []
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            # complete path
            pdf_path = os.path.join(folder_path,filename)
            
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            
            all_docs.extend(docs)
            num_docs += 1
            
    print("Total pdfs :",num_docs)
    print("Total pages :",all_docs)
    return all_docs
