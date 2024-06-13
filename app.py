import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_doc,start=-1,end=-1):
    text=""
    if(start==-1 and end==-1):
        pdf_reader=PdfReader(pdf_doc)
        for page in pdf_reader.pages:
            text+=page.extract_text()
        return text
    pdf_reader=PdfReader(pdf_doc)
    num_pages=len(pdf_reader.pages)
    if(end>=num_pages):
        end=num_pages
    for page in range(start-1,end):
        text+=pdf_reader.pages[page].extract_text()
    print(text)
    return text

def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks=text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversation_chain():
    prompt_template="""
    Answer the question as detailed as possible from the provided context, make sure to provide all the details,
    if the answer is not found in the provided context just say, "Answer is not available in the PDF", don't provide the wrong answer.
    Context:\n {context}\n
    Question:\n{question}\n
    
    Answer:    
    """

    model=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

    prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])
    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain


def user_input(user_question):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db=FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
    docs=new_db.similarity_search(user_question)

    chain=get_conversation_chain()

    response=chain(
        {"input_documents":docs,"question":user_question},
        return_only_outputs=True)
    
    print(response)
    st.write("Reply: ", response["output_text"])
    

def main():
    st.set_page_config("Chat with PDF")
    st.header("Chat with the PDF")
    user_question=st.text_input("Ask a Question from the PDF File")
   


    if user_question:
        user_input(user_question)
    
    with st.sidebar:
        st.title("Menu:")
        start_range=int(st.number_input("Enter the starting page number",value=0,format="%d"))
        end_range=int(st.number_input("Enter the ending page number",value=0,step=1,format="%d"))
        pdf_doc=st.file_uploader("Upload your PDF file and enter the range, then click on the Submit and Process")
        if st.button("Submit and Process"):
            with st.spinner("Processing"):
                if(start_range and end_range):
                    raw_text=get_pdf_text(pdf_doc,start_range,end_range)
                else:
                    raw_text=get_pdf_text(pdf_doc)
                text_chunks=get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
if __name__=="__main__":
    main()