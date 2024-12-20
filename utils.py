from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from langchain_huggingface import  HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForCausalLM

from dotenv import load_dotenv

from PyPDF2 import PdfReader
from docx import Document as py_doc
import os
load_dotenv()

os.environ['GOOGLE_API_KEY']

def create_pipeline(modelname):
    if modelname == 'huggingface':
        model_name = 'deepset/roberta-base-squad2' #"distilbert-base-uncased-distilled-squad" 
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
        return qa_pipeline

    if modelname == 'genai':
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        return llm
    

def create_embeddings(modelname):
    if modelname == 'huggingface':
        model_name = "sentence-transformers/all-mpnet-base-v2"  
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        return embeddings
    if modelname == 'genai':
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        return embeddings

def create_vector_db(docs, embeddings):
    db = FAISS.from_documents(docs, embeddings)
    print("creating vector database")
    return db

def process_file(file_path, file_type):
    content = ""
    if file_type == "pdf":
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                content += page.extract_text()
    elif file_type == "docx":
        doc = py_doc(file_path)
        content = "\n".join([para.text for para in doc.paragraphs])
    elif file_type == "txt":
        with open(file_path, "r") as f:
            content = f.read()
    else:
        raise ValueError("Unsupported file type")
    return content

def split_text(text, chunk_size=1000, overlap=150):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = text_splitter.split_text(text)

    # Converting each chunk into a Document object
    documents = [Document(page_content=chunk) for chunk in chunks]
    print('creating document chunks!')
    return documents
