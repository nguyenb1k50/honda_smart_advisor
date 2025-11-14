import os
import json
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from py2neo import Graph, Node, Relationship




# Load env
load_dotenv()
OPENAI_API_KEY = os.getenv("AZURE_CHAT_OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")
DEPLOYMENT_NAME = os.getenv("AZURE_CHAT_OPENAI_DEPLOYMENT_NAME")

# Neo4j connection
graph = Graph(os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")))

# Ensure Neo4j index for performance
graph.run("""
CREATE CONSTRAINT car_name_unique IF NOT EXISTS
FOR (c:Car) REQUIRE c.name IS UNIQUE
""")

graph.run("""
CREATE CONSTRAINT feature_name_unique IF NOT EXISTS
FOR (f:Feature) REQUIRE f.name IS UNIQUE
""")

graph.run("""
CREATE CONSTRAINT safety_name_unique IF NOT EXISTS
FOR (s:Safety) REQUIRE s.name IS UNIQUE
""")


# LLM init
llm = AzureChatOpenAI(
    api_key=OPENAI_API_KEY,
    azure_endpoint=OPENAI_API_BASE,
    deployment_name=DEPLOYMENT_NAME,
    openai_api_version="2023-07-01-preview",
    temperature=0
)

# === Define structured schema ===
class NodeSchema(BaseModel):
    type: str
    properties: dict

class RelationshipSchema(BaseModel):
    start_node: str
    end_node: str
    type: str

class GraphSchema(BaseModel):
    nodes: list[NodeSchema] = Field(default_factory=list)
    relationships: list[RelationshipSchema] = Field(default_factory=list)

parser = PydanticOutputParser(pydantic_object=GraphSchema)

# === Step 1: Extract text from PDFs with chunking ===
def extract_text_chunks(pdf_files, chunk_size=2000, chunk_overlap=10):
    all_docs = []
    for pdf_file in pdf_files:
        loader = UnstructuredPDFLoader(pdf_file)
        docs = loader.load()
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(all_docs)

# === Step 2: Use LLM to parse structured info ===
def parse_graph_with_llm(chunks):
    all_nodes = []
    all_relationships = []

    for i, chunk in enumerate(chunks):
        prompt = f"""
        Bạn là hệ thống trích xuất thông tin từ tài liệu PDF về xe Honda.
        Hãy phân tích nội dung sau và trả về dữ liệu dưới dạng JSON với cấu trúc sau:

        {parser.get_format_instructions()}

        Nội dung:
        {chunk.page_content}
        """
        response = llm.invoke(prompt)
        try:
            graph_data = parser.parse(response)
            all_nodes.extend(graph_data.nodes)
            all_relationships.extend(graph_data.relationships)
        except Exception as e:
            print(f"[Chunk {i}] Lỗi parse JSON: {e}")

    return {"nodes": all_nodes, "relationships": all_relationships}

# === Step 3: Load vào Neo4j ===
def load_graph_to_neo4j(graph_data):
    node_map = {}
    for node in graph_data["nodes"]:
        n = Node(node.type, **node.properties)
        graph.merge(n, node.type, "name")
        node_map[f"{node.type}:{node.properties['name']}"] = n

    for rel in graph_data["relationships"]:
        start = node_map.get(rel.start_node)
        end = node_map.get(rel.end_node)
        if start and end:
            r = Relationship(start, rel.type, end)
            graph.merge(r)

# === Main ETL ===
if __name__ == "__main__":
    pdf_files = ["Camry.pdf", "crv.pdf"]
    chunks = extract_text_chunks(pdf_files)
    graph_data = parse_graph_with_llm(chunks)
    load_graph_to_neo4j(graph_data)
