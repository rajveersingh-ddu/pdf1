import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from streamlit_navigation_bar import st_navbar

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))






def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])




def main():
    st.set_page_config(initial_sidebar_state="collapsed")
   
    pages = ["🏘 Home", "Library", "Tutorials", "Development", "Download"]
    styles = {
    "nav": {
        "background-color": "#060270",
    },
    "div": {
        "max-width": "32rem",
    },
    "span": {
        "border-radius": "0.5rem",
        "padding": "0.4375rem 0.625rem",
        "margin": "0 0.125rem",
    },
    "active": {
        "background-color": "rgba(255, 255, 255, 0.50)",
    },
    "hover": {
        "background-color": "rgba(255, 255, 255, 0.35)",
    },
    }
    page = st_navbar(pages, styles=styles)
    html_string = "<h1>Ask About Pdf</h1>"
    st.markdown(html_string, unsafe_allow_html=True)
    

    
    st.subheader("Ask a question about your PDF")
    pdf_docs = st.file_uploader("For processing, upload your PDF files and click the 'Submit & Process' button.", accept_multiple_files=True)
    if st.button("Start Processing!"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Done")
    st.header("Talk to your PDFs with the power of AI!💁")
    user_question = st.text_input("Have a question about your document? Ask here!")
    # Create a container for the footer
    st.divider()
    footer = st.container()

# Set the desired styling for the footer (optional)
    with footer:
     st.markdown(""" <style>
        .footer {
            position: relative;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #fcba03;
            color: #fcba03;
            text-align: center;
            padding: 10px;
        }
        </style> """, unsafe_allow_html=True)
# Add content to the footer
    with footer:
     st.write("Developed with ❤️ ")
     st.write("© 2024 All rights reserved.")
    



    if user_question:
        user_input(user_question)

    
   


if __name__ == "__main__":
    main()
