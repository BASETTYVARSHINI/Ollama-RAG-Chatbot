
# Step 2: Import required libraries
from langchain_community.chat_models import ChatOllama
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA  
from PyPDF2 import PdfReader
from langchain.schema import Document

# Step 3: Function to load and extract text from the PDF
def extract_text_from_pdf(pdf_path):
    try:
        pdf_reader = PdfReader(pdf_path)
        text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        if not text:
            raise ValueError("No text found in the PDF!")
        return text
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return None

# Step 4: Function to initialize the language model
def initialize_llm(model_name="llama3.1"):
    return ChatOllama(model=model_name, keep_alive="3h", temperature=0)

# Step 5: Function to process text and create vector embeddings
def create_vector_store(text):
    docs = [Document(page_content=text, metadata={"source": "faculty_information"})]
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    chunked_documents = text_splitter.split_documents(docs)

    print(f"Total Chunks Created: {len(chunked_documents)}")

    embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=False)
    db = FAISS.from_documents(chunked_documents, embeddings)
    return db

# Step 6: Function to set up the RAG model
def setup_qa_system(vector_store, llm):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 4})
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# Step 7: Function to query the chatbot
def ask_question(qa_system, question):
    response = qa_system({"query": question})
    return response["result"]

# ---- MAIN EXECUTION ----
if __name__ == "__main__":
    print("WELCOME TO VIT ASSIST CHATBOT")

    # Load PDF and initialize components
    pdf_path = "data/VITC_FACULTY.pdf"
    extracted_text = extract_text_from_pdf(pdf_path)

    if extracted_text:
        llm = initialize_llm()
        vector_store = create_vector_store(extracted_text)
        qa_system = setup_qa_system(vector_store, llm)

        # Example Queries
        questions = ["HELLO", "Sreedevi V T has cabin detail", "What is the location of VIT Chennai?"]
        for q in questions:
            print(f"\nUSER: {q}")
            print(f"BOT: {ask_question(qa_system, q)}")
    else:
        print("Chatbot setup failed due to PDF extraction error.")
