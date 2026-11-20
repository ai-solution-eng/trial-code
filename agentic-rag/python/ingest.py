import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

#==============================================================================

load_dotenv()

EMBED_URL   = os.environ.get("EMBED_URL",   "")
EMBED_KEY   = os.environ.get("EMBED_KEY",   "")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")

QDRANT_URL        = os.environ.get("QDRANT_URL",        "http://qdrant.qdrant.svc.cluster.local")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "documents")

DATA_DIR          = Path(os.environ.get("DATA_DIR",      "./data"))
CHUNK_SIZE        = int (os.environ.get("CHUNK_SIZE",    "1000"))
CHUNK_OVERLAP     = int (os.environ.get("CHUNK_OVERLAP", "150"))

LOADERS = {
    ".txt": TextLoader,
    ".md" : TextLoader,
    ".pdf": PyPDFLoader,
}

#==============================================================================

def ingest_document(doc: list[Document], embeddings: NVIDIAEmbeddings) -> None:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(doc)

    QdrantVectorStore.from_documents(
        chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=QDRANT_COLLECTION,
        prefer_grpc=False,
    )

#==============================================================================

def ingest_documents(data_dir: Path, embeddings: NVIDIAEmbeddings) -> None:
    files = sorted(path for path in data_dir.rglob("*") if path.is_file() and path.suffix.lower() in LOADERS)

    for path in files:
        loader_cls = LOADERS[path.suffix.lower()]
        loader = (
            loader_cls(str(path), encoding="utf-8") if loader_cls is TextLoader 
            else loader_cls(str(path))
        )

        doc = loader.load()
        ingest_document(doc, embeddings)
        print(f"Loaded {path} ({len(doc)} page(s))")

#==============================================================================

def main() -> int:
    client     = QdrantClient(url=QDRANT_URL, prefer_grpc=False, timeout=30)
    embeddings = NVIDIAEmbeddings(model=EMBED_MODEL, base_url=EMBED_URL, api_key=EMBED_KEY, truncate="END")

    if not client.collection_exists(QDRANT_COLLECTION):
        print(f"Creating collection '{QDRANT_COLLECTION}'.")
        client.create_collection(
            QDRANT_COLLECTION,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )

    ingest_documents(DATA_DIR, embeddings)

    print("Done.")
    return 0

#==============================================================================

if __name__ == "__main__":
    sys.exit(main())

#==============================================================================
