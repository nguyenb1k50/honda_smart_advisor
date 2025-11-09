import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureOpenAIEmbeddings

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# === Step 1: Load and split PDF documents ===
def load_and_split_pdfs(pdf_files):
    all_documents = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    for pdf_file in pdf_files:
        loader = UnstructuredPDFLoader(pdf_file)
        docs = loader.load()
        print(f"Loaded {len(docs)} pages from {pdf_file}")
        split_docs = splitter.split_documents(docs)
        print(f"Split into {len(split_docs)} chunks")
        all_documents.extend(split_docs)
    return all_documents

# === Step 2: Create embeddings and store in ChromaDB ===
def create_chroma_vectorstore(documents, persist_directory="./honda_chroma"):
    embedding_model = AzureOpenAIEmbeddings(
    azure_deployment=DEPLOYMENT_NAME,
    openai_api_key=OPENAI_API_KEY,
    azure_endpoint=OPENAI_API_BASE,
    openai_api_version="2023-05-15"
    )
    
    if not documents:
        raise ValueError("Không có tài liệu nào được load hoặc tách. Kiểm tra file PDF và splitter.")

    vectorstore = Chroma.from_documents(documents=documents, embedding=embedding_model, persist_directory=persist_directory)
    vectorstore.persist()
    return vectorstore

# === Step 3: Create RAG chain manually ===
def create_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = AzureChatOpenAI(
        api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_API_BASE,
        deployment_name=DEPLOYMENT_NAME,
        openai_api_type="azure",
        openai_api_version="2023-07-01-preview"
    )
    prompt = PromptTemplate.from_template("Dựa trên thông tin sau đây: {context} Hãy trả lời câu hỏi: {question}"
    )
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# === Step 4: Define Function Calling ===
def get_car_info(rag_chain, car_name):
    return rag_chain.invoke(f"Thông tin chi tiết về xe {car_name}")

def compare_cars(rag_chain, car1, car2):
    info1 = get_car_info(rag_chain, car1)
    info2 = get_car_info(rag_chain, car2)
    return f"So sánh giữa {car1} và {car2}: {car1}: {info1} {car2}: {info2}"

def recommend_car(rag_chain, criteria):
    return rag_chain.invoke(f"Tôi cần xe phù hợp với tiêu chí: {criteria}")

# === Step 5: Streamlit Frontend ===
def run_streamlit_app(rag_chain):
    st.set_page_config(page_title="Honda Smart Advisor", layout="wide")
    st.title("🚗 Honda Smart Advisor Chatbot")

    option = st.selectbox("Chọn chức năng", ["Tư vấn xe", "So sánh xe", "Đề xuất xe"])

    if option == "Tư vấn xe":
        car_name = st.text_input("Nhập tên xe bạn muốn tìm hiểu:")
        if st.button("Xem thông tin"):
            if car_name:
                result = get_car_info(rag_chain, car_name)
                st.write(result)

    elif option == "So sánh xe":
        car1 = st.text_input("Xe thứ nhất:")
        car2 = st.text_input("Xe thứ hai:")
        if st.button("So sánh"):
            if car1 and car2:
                result = compare_cars(rag_chain, car1, car2)
                st.write(result)

    elif option == "Đề xuất xe":
        criteria = st.text_area("Nhập tiêu chí (ví dụ: giá dưới 900 triệu, tiết kiệm nhiên liệu, có hỗ trợ an toàn):")
        if st.button("Đề xuất"):
            if criteria:
                result = recommend_car(rag_chain, criteria)
                st.write(result)

# === Main Execution ===
if __name__ == "__main__":
    pdf_files = ["city.pdf", "crv.pdf"]
    documents = load_and_split_pdfs(pdf_files)
    vectorstore = create_chroma_vectorstore(documents)
    rag_chain = create_rag_chain(vectorstore)
    run_streamlit_app(rag_chain)
