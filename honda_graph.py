import os
import json
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from py2neo import Graph, Node, Relationship
from langchain_text_splitters import TokenTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph





# Load env
load_dotenv()
OPENAI_API_KEY = os.getenv("AZURE_CHAT_OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")
DEPLOYMENT_NAME = os.getenv("AZURE_CHAT_OPENAI_DEPLOYMENT_NAME")

# Neo4j connection
graph = Graph(os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")))

# Ensure Neo4j index for performance
# graph.run("""
# CREATE CONSTRAINT car_name_unique IF NOT EXISTS
# FOR (c:Car) REQUIRE c.name IS UNIQUE
# """)

# graph.run("""
# CREATE CONSTRAINT feature_name_unique IF NOT EXISTS
# FOR (f:Feature) REQUIRE f.name IS UNIQUE
# """)

# graph.run("""
# CREATE CONSTRAINT safety_name_unique IF NOT EXISTS
# FOR (s:Safety) REQUIRE s.name IS UNIQUE
# """)


# LLM init
llm = AzureChatOpenAI(
    api_key=OPENAI_API_KEY,
    azure_endpoint=OPENAI_API_BASE,
    deployment_name=DEPLOYMENT_NAME,
    openai_api_version="2023-07-01-preview",
    temperature=0
)

llm_transformer = LLMGraphTransformer(llm=llm)

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
def extract_text_chunks(pdf_files, chunk_size=1000, chunk_overlap=20):
    documents = []
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
   
    for pdf_file in pdf_files:
        loader = UnstructuredPDFLoader(pdf_file)
        docs = loader.load()

        chunks = text_splitter.split_documents(docs)
        
        for chunk in chunks:
            chunk.metadata["source"] = pdf_file
        
        documents.extend(chunks)
    print(f"Total chunks: {len(documents)}")
    # splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return documents

# === Step 2: Use LLM to parse structured info ===
# def parse_graph_with_llm(chunks):
#     all_nodes = []
#     all_relationships = []

#     for i, chunk in enumerate(chunks):
#         prompt = f"""
#         Bạn là hệ thống trích xuất thông tin từ tài liệu PDF về xe Honda.
#         Hãy phân tích nội dung sau và trả về dữ liệu dưới dạng JSON với cấu trúc sau:

#         {parser.get_format_instructions()}

#         Nội dung:
#         {chunk.page_content}
#         """
#         response = llm.invoke(prompt)
#         try:
#             graph_data = parser.parse(response)
#             all_nodes.extend(graph_data.nodes)
#             all_relationships.extend(graph_data.relationships)
#         except Exception as e:
#             print(f"[Chunk {i}] Lỗi parse JSON: {e}")

#     return {"nodes": all_nodes, "relationships": all_relationships}


# ====Step 2 update

def parse_graph_data(chunks):
    graph_documents = llm_transformer.convert_to_graph_documents(chunks)
    print("graph_documents", graph_documents)
    return graph_documents
# === Step 3: Load vào Neo4j ===

# def load_graph_to_neo4j(graph_data):
#     all_nodes = []
#     all_relationships = []
#     for item in graph_data:
#         all_nodes.extend(item.nodes)
#         all_relationships.extend(item.relationships)

#     node_map = {}
#     for node in all_nodes:
#         n = Node(node.type, **node.properties)
#         graph.merge(n, node["type"], "name")
#         node_map[f"{node.type}:{node.properties['name']}"] = n
#     for rel in all_relationships:
#         start = node_map.get(rel["start_node"])
#         end = node_map.get(rel["end_node"])
#         if start and end:
#             r = Relationship(start, rel["type"], end)
#             graph.merge(r)
from py2neo import Graph, Node, Relationship

def load_graph_to_neo4j(graph_data, graph):
    # Mock data (có thể thay bằng graph_data thực tế)
    all_nodes = []

    all_relationships = []

    for item in graph_data:
        all_nodes.extend(item["nodes"])
        all_relationships.extend(item["relationships"])

    # Tạo node bằng Cypher MERGE
    for node in all_nodes:
        node_type = node.get("type")
        properties = node.get("properties", {})
        if not node_type or "name" not in properties:
            print(f"⚠ Node thiếu thông tin: {node}")
            continue

        # Tạo câu lệnh Cypher động
        set_clause = ", ".join([f"n.{k} = ${k}" for k in properties.keys() if k != "name"])
        cypher = f"""
        MERGE (n:{node_type} {{name: $name}})
        SET {set_clause}
        """
        graph.query(cypher, params=properties)

    # Tạo relationship bằng Cypher MERGE
    for rel in all_relationships:
        cypher_rel = f"""
        MATCH (a:{rel['source_node_type']} {{name: $source_name}})
        MATCH (b:{rel['target_node_type']} {{name: $target_name}})
        MERGE (a)-[r:{rel['type']}]->(b)
        """
        graph.query(cypher_rel, params={
            "source_name": rel["source_node_id"],
            "target_name": rel["target_node_id"]
        })

    print("✅ Graph đã được load vào Neo4j thành công!")

# === Main ETL ===
if __name__ == "__main__":
    pdf_files = ["Camry.pdf", "crv.pdf"]
    # chunks = extract_text_chunks(pdf_files)
    # graph_data = parse_graph_data(chunks)
        
    graph_data = [
        {
            "nodes": [
                {"type": "CarModel", "properties": {"name": "Honda CR-V", "brand": "Honda", "segment": "SUV"}},
                {"type": "Specification", "properties": {"name": "Engine", "value": "1.5L VTEC Turbo"}},
                {"type": "Specification", "properties": {"name": "Transmission", "value": "CVT"}},
                {"type": "Specification", "properties": {"name": "FuelType", "value": "Petrol"}},
                {"type": "Feature", "properties": {"name": "Safety", "value": "Honda Sensing"}},
                {"type": "Feature", "properties": {"name": "Infotainment", "value": "7-inch touchscreen"}}
            ],
            "relationships": [
                {"source_node_id": "Honda CR-V", "source_node_type": "CarModel", "target_node_id": "Engine", "target_node_type": "Specification", "type": "has_spec"},
                {"source_node_id": "Honda CR-V", "source_node_type": "CarModel", "target_node_id": "Transmission", "target_node_type": "Specification", "type": "has_spec"},
                {"source_node_id": "Honda CR-V", "source_node_type": "CarModel", "target_node_id": "FuelType", "target_node_type": "Specification", "type": "has_spec"},
                {"source_node_id": "Honda CR-V", "source_node_type": "CarModel", "target_node_id": "Safety", "target_node_type": "Feature", "type": "has_feature"},
                {"source_node_id": "Honda CR-V", "source_node_type": "CarModel", "target_node_id": "Infotainment", "target_node_type": "Feature", "type": "has_feature"}
            ]
        },
        {
            "nodes": [
                {"type": "CarModel", "properties": {"name": "Honda Civic", "brand": "Honda", "segment": "Sedan"}},
                {"type": "Specification", "properties": {"name": "Engine", "value": "2.0L i-VTEC"}},
                {"type": "Specification", "properties": {"name": "Transmission", "value": "6-speed manual"}},
                {"type": "Specification", "properties": {"name": "FuelType", "value": "Petrol"}},
                {"type": "Feature", "properties": {"name": "Safety", "value": "Airbags"}},
                {"type": "Feature", "properties": {"name": "Infotainment", "value": "Apple CarPlay"}}
            ],
            "relationships": [
                {"source_node_id": "Honda Civic", "source_node_type": "CarModel", "target_node_id": "Engine", "target_node_type": "Specification", "type": "has_spec"},
                {"source_node_id": "Honda Civic", "source_node_type": "CarModel", "target_node_id": "Transmission", "target_node_type": "Specification", "type": "has_spec"},
                {"source_node_id": "Honda Civic", "source_node_type": "CarModel", "target_node_id": "FuelType", "target_node_type": "Specification", "type": "has_spec"},
                {"source_node_id": "Honda Civic", "source_node_type": "CarModel", "target_node_id": "Safety", "target_node_type": "Feature", "type": "has_feature"},
                {"source_node_id": "Honda Civic", "source_node_type": "CarModel", "target_node_id": "Infotainment", "target_node_type": "Feature", "type": "has_feature"}
            ]
        },
        {
            "nodes": [
                {"type": "CarModel", "properties": {"name": "Honda Accord", "brand": "Honda", "segment": "Sedan"}},
                {"type": "Specification", "properties": {"name": "Engine", "value": "2.4L DOHC i-VTEC"}},
                {"type": "Specification", "properties": {"name": "Transmission", "value": "8-speed automatic"}},
                {"type": "Specification", "properties": {"name": "FuelType", "value": "Hybrid"}},
                {"type": "Feature", "properties": {"name": "Safety", "value": "Honda Sensing"}},
                {"type": "Feature", "properties": {"name": "Infotainment", "value": "8-inch touchscreen"}}
            ],
            "relationships": [
                {"source_node_id": "Honda Accord", "source_node_type": "CarModel", "target_node_id": "Engine", "target_node_type": "Specification", "type": "has_spec"},
                {"source_node_id": "Honda Accord", "source_node_type": "CarModel", "target_node_id": "Transmission", "target_node_type": "Specification", "type": "has_spec"},
                {"source_node_id": "Honda Accord", "source_node_type": "CarModel", "target_node_id": "FuelType", "target_node_type": "Specification", "type": "has_spec"},
                {"source_node_id": "Honda Accord", "source_node_type": "CarModel", "target_node_id": "Safety", "target_node_type": "Feature", "type": "has_feature"},
                {"source_node_id": "Honda Accord", "source_node_type": "CarModel", "target_node_id": "Infotainment", "target_node_type": "Feature", "type": "has_feature"}
            ]
        },
        {
            "nodes": [
                {"type": "CarModel", "properties": {"name": "Honda City", "brand": "Honda", "segment": "Sedan"}},
                {"type": "Specification", "properties": {"name": "Engine", "value": "1.5L i-VTEC"}},
                {"type": "Specification", "properties": {"name": "Transmission", "value": "CVT"}},
                {"type": "Specification", "properties": {"name": "FuelType", "value": "Petrol"}},
                {"type": "Feature", "properties": {"name": "Safety", "value": "ABS"}},
                {"type": "Feature", "properties": {"name": "Infotainment", "value": "7-inch touchscreen"}}
            ],
            "relationships": [
                {"source_node_id": "Honda City", "source_node_type": "CarModel", "target_node_id": "Engine", "target_node_type": "Specification", "type": "has_spec"},
                {"source_node_id": "Honda City", "source_node_type": "CarModel", "target_node_id": "Transmission", "target_node_type": "Specification", "type": "has_spec"},
                {"source_node_id": "Honda City", "source_node_type": "CarModel", "target_node_id": "FuelType", "target_node_type": "Specification", "type": "has_spec"},
                {"source_node_id": "Honda City", "source_node_type": "CarModel", "target_node_id": "Safety", "target_node_type": "Feature", "type": "has_feature"},
                {"source_node_id": "Honda City", "source_node_type": "CarModel", "target_node_id": "Infotainment", "target_node_type": "Feature", "type": "has_feature"}
            ]
        },
        {
            "nodes": [
                {"type": "CarModel", "properties": {"name": "Honda HR-V", "brand": "Honda", "segment": "SUV"}},
                {"type": "Specification", "properties": {"name": "Engine", "value": "1.8L i-VTEC"}},
                {"type": "Specification", "properties": {"name": "Transmission", "value": "CVT"}},
                {"type": "Specification", "properties": {"name": "FuelType", "value": "Petrol"}},
                {"type": "Feature", "properties": {"name": "Safety", "value": "Honda Sensing"}},
                {"type": "Feature", "properties": {"name": "Infotainment", "value": "Apple CarPlay"}}
            ],
            "relationships": [
                {"source_node_id": "Honda HR-V", "source_node_type": "CarModel", "target_node_id": "Engine", "target_node_type": "Specification", "type": "has_spec"},
                {"source_node_id": "Honda HR-V", "source_node_type": "CarModel", "target_node_id": "Transmission", "target_node_type": "Specification", "type": "has_spec"},
                {"source_node_id": "Honda HR-V", "source_node_type": "CarModel", "target_node_id": "FuelType", "target_node_type": "Specification", "type": "has_spec"},
                {"source_node_id": "Honda HR-V", "source_node_type": "CarModel", "target_node_id": "Safety", "target_node_type": "Feature", "type": "has_feature"},
                {"source_node_id": "Honda HR-V", "source_node_type": "CarModel", "target_node_id": "Infotainment", "target_node_type": "Feature", "type": "has_feature"}
            ]
        }
    ]
    graph = Neo4jGraph(url=os.getenv("NEO4J_URI"), username=os.getenv("NEO4J_USER"), password=os.getenv("NEO4J_PASSWORD")
)
    # graph_data = parse_graph_with_llm(chunks)
    load_graph_to_neo4j(graph_data, graph)
