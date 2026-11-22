import json
import os

from mcp.server.fastmcp import FastMCP

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

#==============================================================================

load_dotenv()

EMBED_URL   = os.environ.get("EMBED_URL",   "")
EMBED_KEY   = os.environ.get("EMBED_KEY",   "")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")

QDRANT_URL        = os.environ.get("QDRANT_URL",        "http://qdrant.qdrant.svc.cluster.local")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "documents")
TOP_K             = int(os.environ.get("TOP_K",         "4"))

HOST = os.getenv("MCP_HOST", "0.0.0.0")
PORT = int(os.getenv("MCP_PORT", "8000"))

#==============================================================================

embeddings = NVIDIAEmbeddings(model=EMBED_MODEL, base_url=EMBED_URL, api_key=EMBED_KEY, truncate="END")
client     = QdrantClient(url=QDRANT_URL, prefer_grpc=False, timeout=30)
store      = QdrantVectorStore(client=client, collection_name=QDRANT_COLLECTION, embedding=embeddings)
mcp        = FastMCP("vector-search", stateless_http=True, json_response=True, host=HOST, port=PORT)

#==============================================================================

@mcp.tool()
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base and return the most relevant document chunks.

    Args:
        query: Natural-language search query.
    """
    results = store.similarity_search_with_score(query, k=TOP_K)
    hits = [
        {"content": doc.page_content, "score": score, "metadata": doc.metadata} for doc, score in results
    ]
    return json.dumps({"query": query, "results": hits}, ensure_ascii=False)

#==============================================================================

if __name__ == "__main__":
    mcp.run(transport="streamable-http")

#==============================================================================
