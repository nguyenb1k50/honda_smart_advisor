import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_experimental.graph_transformers import LLMGraphTransformer

# Load env
load_dotenv()
OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# === Step 1: Kết nối Neo4j ===
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD")
)

# === Step 2: Khởi tạo LLM ===
llm = AzureChatOpenAI(
    api_key=OPENAI_API_KEY,
    azure_endpoint=OPENAI_API_BASE,
    deployment_name=DEPLOYMENT_NAME,
    openai_api_version="2023-07-01-preview",
    temperature=0
)

# === Step 3: Cypher Translator + Graph QA Chain ===
graph_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True  # Cho phép query phức tạp
)

# === Step 4: LLMGraphTransformer ===
graph_transformer = LLMGraphTransformer(llm=llm)

# === Step 5: Hàm xử lý câu hỏi ===

def ask_question(question):
    # Tạo Cypher query và lấy kết quả từ Neo4j
    raw_result = graph_chain.run(question)
    # Biến đổi dữ liệu graph thành câu trả lời tự nhiên
    return graph_transformer(raw_result)



# === Step 6: Streamlit UI ===
def run_streamlit_app():
    st.set_page_config(page_title="Honda Smart Advisor (Graph)", layout="wide")
    st.title("🚗 Honda Smart Advisor - Graph RAG")

    st.write("Hỏi bất kỳ về xe Honda (ví dụ: 'Xe nào dưới 700 triệu có ABS?')")

    question = st.text_input("Nhập câu hỏi:")
    if st.button("Hỏi"):
        if question:
            with st.spinner("Đang xử lý..."):
                answer = ask_question(question)
                st.write(answer)

    st.markdown("### Ví dụ câu hỏi:")
    st.markdown("- Xe nào dưới 700 triệu có tính năng ABS?")
    st.markdown("- So sánh Honda City và Honda CR-V")
    st.markdown("- Đề xuất xe tiết kiệm nhiên liệu và có hỗ trợ an toàn")

if __name__ == "__main__":
    run_streamlit_app()