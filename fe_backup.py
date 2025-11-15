import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_experimental.graph_transformers import LLMGraphTransformer
# from langchain.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate

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

# === Step 4: LLMGraphTransformer ===
graph_transformer = LLMGraphTransformer(llm=llm)

# 3. Prompt Template với fallback
prompt_template = PromptTemplate(
    input_variables=["context", "query", "question"],
    template="""
Bạn là chuyên gia tư vấn xe Honda. Dựa trên dữ liệu sau:
{context}

Câu hỏi: {question}

Nếu không tìm thấy thông tin, trả lời: "Hiện tại tôi chưa có dữ liệu về xe này."
"""
)


# 4. Tạo GraphChain
graph_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    cypher_prompt=PromptTemplate(
        input_variables=["question"],
        template="""
Viết câu lệnh Cypher để trả lời câu hỏi sau:
{question}
Chỉ trả về dữ liệu cần thiết, không thêm giải thích.
"""
    ),
    qa_prompt=prompt_template,
    verbose=True,
    allow_dangerous_requests=True
)



def ask_question(question, graph_chain, llm):
    """
    Truy vấn Neo4j bằng GraphCypherQAChain.
    Nếu không có dữ liệu hoặc trả về "I don't know", fallback sang LLM để trả lời dựa trên kiến thức chung.
    """
    print(f"📌 Question: {question}")

    try:
        # Gọi LangChain chain để thực hiện truy vấn Neo4j
        raw_result = graph_chain.invoke({"query": question})
        print("🔍 Raw result:", raw_result)

        # Nếu không có kết quả hoặc LLM trả về "I don't know"
        # if not raw_result or raw_result.get("result") in ["I don't know the answer.", "Tôi không biết câu trả lời."]:
        #     print("⚠ Không tìm thấy dữ liệu trong Neo4j, fallback sang LLM...")
        #     # Fallback: dùng LLM để trả lời dựa trên kiến thức chung
        #     llm_response = llm.invoke(question)
        #     return llm_response.content
        if not raw_result or raw_result.get("result") in ["I don't know the answer.", "Hiện tại tôi chưa có dữ liệu về xe này."]:
            print("⚠ Không tìm thấy dữ liệu trong Neo4j, fallback sang LLM...")
            # Fallback: dùng LLM để trả lời dựa trên kiến thức chung
            llm_response = llm.invoke(question)
            return "Không tìm thấy dữ liệu trong hệ thống, nhưng tôi sẽ cung cấp cho bạn thông tin sau: \n\n \n " + llm_response.content

        # Trả về câu trả lời tự nhiên từ LLM trong chain
        return raw_result.get("result", "Không có câu trả lời.")

    except Exception as e:
        print(f"⚠ Lỗi khi xử lý câu hỏi: {e}")
        return "Đã xảy ra lỗi khi truy vấn dữ liệu."


# === Step 6: Streamlit UI ===
def run_streamlit_app():
    st.set_page_config(page_title="Honda Smart Advisor (Graph)", layout="wide")
    st.title("🚗 Honda Smart Advisor - Graph RAG")

    st.write("Hỏi bất kỳ về xe Honda (ví dụ: 'Xe nào dưới 700 triệu có ABS?')")

    question = st.text_input("Nhập câu hỏi:")
    if st.button("Hỏi"):
        if question:
            with st.spinner("Đang xử lý..."):
                answer = ask_question(question, graph_chain, llm)
                st.write(answer)

    st.markdown("### Ví dụ câu hỏi:")
    st.markdown("- Xe nào dưới 700 triệu có tính năng ABS?")
    st.markdown("- So sánh Honda City và Honda CR-V")
    st.markdown("- Đề xuất xe tiết kiệm nhiên liệu và có hỗ trợ an toàn")

if __name__ == "__main__":
    run_streamlit_app()