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
    st.set_page_config(layout="wide",initial_sidebar_state="collapsed")
    
    st.sidebar.title("PolyReader")
    st.sidebar.image('logo.jpeg')
    st.sidebar.write("Comming Soon")

    pages = ["🏘 Home", "Library", "Tutorials", "Development", "Download"]
    styles = {
    "nav": {
        "background-color": "#567002",
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
        "background-color": "rgba(8,37,81, 0.50)",
    },
    "hover": {
        "background-color": "rgba(8, 37, 81, 0.25)",
    },
    }
    page = st_navbar(pages, styles=styles)
    # adding the logo
    # Create a column layout
    col1, col2,col3 = st.columns([1, 1,1])

# Add text in the first column
    with col1:
      st.write("")

# Add image in the second column
    with col2:
       st.image("logo.jpeg",width=300)
#Add text in third Column
    with col1:
       st.write("")
    

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
    

    #center

    st.markdown("""
Imagine a world where your PDFs are no longer static walls of text. AskYourPDF, powered by ChatGPT, transforms them into dynamic companions, ready to answer your questions and reveal hidden insights.
""")

    col1, col2 = st.columns(2)

    with col1:
     st.subheader("Stop Scrolling, Start Asking")
     st.write("Forget aimlessly searching through pages. With PolyReader, simply upload your document and ask a question.  It could be anything:")
     st.write("* What are the key takeaways from this contract?")
     st.write("* Summarize the financial performance for Q3.")
     st.write("* Find all mentions of competitor analysis.")

    with col2:
     st.subheader("Uncover Deeper Understandings")
     st.write("Go beyond basic retrieval. polyReader can analyze the context of your questions, drawing connections and highlighting relevant passages. It doesn't just provide answers, it fosters a deeper understanding of the content.")

     st.subheader("Boost Engagement and Efficiency")
     st.write("""
Whether you're a student deciphering a complex research paper or a professional navigating a lengthy report, PolyReader empowers you to interact with documents in a whole new way. Save time, improve comprehension, and unlock the full potential of your PDFs.
""")


    
    # Create a container for the footer
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
    footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
left: 0;
bottom: 0;
width: 100%;
background-color: teal;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with ❤ by<br> Rajveer Singh and Abhishek Kumar</a></p>
</div>
"""
    st.markdown(footer,unsafe_allow_html=True)
    



    if user_question:
        user_input(user_question)

    
   


if __name__ == "__main__":
    main()