#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain.document_loaders import OnlinePDFLoader
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain import PromptTemplate
import os
import nltk
import openai

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

texts = ""
        
openai.api_key ="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"

def input_pdf(texts):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    chunks = text_splitter.create_documents([texts])
    return chunks

def generate_questions_from_chunks(chunks):
    question = []
    for chunk in chunks:
        prompt = """generate one question and the corresponding answer from the chunk,
        the question must satisfy the following rules: 1. A person must have a thorough understanding of the content of the chunk to answer correctly,
        2. The question should only ask about information related to the content of the chunk, 3. The answer must be within 20 words.
        The chunk : {}""".format(str(chunk.page_content))
        response = call_gpt(prompt)
        question.append(response["choices"][0]["message"]["content"])
    return question
        
def call_gpt(prompt):
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-16k",
    messages = [
        {"role": "system", "content": 'do not be too focus on the detail, try to be general'},
        {"role": "user", "content": prompt}
    ]
    )
    return response

if uploaded_file is not None:
    pdf_reader = PdfReader(uploaded_file)
    num_pages = len(pdf_reader.pages)

    for page_number in range(num_pages):
        page = pdf_reader.pages[page_number]
        text = page.extract_text()
        texts = texts + text
    chunks = input_pdf(texts)
    questions = generate_questions_from_chunks(chunks)        
    st.write("success")
    st.write(questions)

