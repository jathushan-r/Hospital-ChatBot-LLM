import os
from typing import Optional

import json
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import CSVLoader



CHROMA_DB_DIRECTORY=r'C:\Users\Jathushan\Documents\DS project\end-to-end-llm-main\src\db'
DOCUMENT_SOURCE_DIRECTORY = r'C:\Users\Jathushan\Documents\DS project\end-to-end-llm-main\src\source_documents'
CHROMA_SETTINGS = Settings(
    chroma_db_impl='duckdb+parquet',
    persist_directory=CHROMA_DB_DIRECTORY,
    anonymized_telemetry=False
)
TARGET_SOURCE_CHUNKS=4
CHUNK_SIZE=100
CHUNK_OVERLAP=30
HIDE_SOURCE_DOCUMENTS=False

class MyKnowledgeBase:
    
    def __init__(self, pdf_source_folder_path: str, vector_db) -> None:
        """
        Loads pdf and creates a Knowledge base using the Chroma
        vector DB.
        Args:
            pdf_source_folder_path (str): The source folder containing 
            all the pdf documents
        """
        self.vector_db = vector_db
        self.pdf_source_folder_path = pdf_source_folder_path

    def get_vector_db(self):
        return self.vector_db


    def load_pdfs(self):
        text_loader_kwargs={'autodetect_encoding': True}
        loader = DirectoryLoader(DOCUMENT_SOURCE_DIRECTORY, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
        docs = loader.load()
        return docs

    def split_documents(
        self,
        loaded_docs,
    ):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "},", " "]
        )
        chunked_docs = splitter.split_documents(loaded_docs)
        return chunked_docs

    def convert_document_to_embeddings(
        self, chunked_docs, embedder
    ):
        vector_db = Chroma.from_documents(chunked_docs, embedder, persist_directory=CHROMA_DB_DIRECTORY)

        vector_db.add_documents(chunked_docs)
        vector_db.persist()
        return vector_db

    def return_retriever_from_persistant_vector_db(
        self
    ):

        return self.vector_db.as_retriever(
            search_kwargs={"k": TARGET_SOURCE_CHUNKS}
        )

    def initiate_document_injetion_pipeline(self):
        loaded_pdfs = self.load_pdfs()
        chunked_documents = self.split_documents(loaded_docs=loaded_pdfs)
        
        print("=> PDF loading and chunking done.")

        embeddings = GPT4AllEmbeddings()
        self.vector_db = self.convert_document_to_embeddings(
            chunked_docs=chunked_documents, embedder=embeddings
        )

        print("=> vector db initialised and created.")
        print("All done")


        

# if __name__ == '__main__':
#     print('Hello')
#     loader = PyPDFDirectoryLoader(r'C:\Users\Jathushan\Documents\end-to-end-llm-main\src\source_documents')
#     loaded_pdfs = loader.load()
