import os
import sys
from pathlib import Path

import boto3
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
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "anywhere")

S3_URL            = os.environ.get("S3_URL",    "http://local-s3-service.ezdata-system.svc.cluster.local:30000")
S3_BUCKET         = os.environ.get("S3_BUCKET", "documents")
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

def ingest_documents(embeddings: NVIDIAEmbeddings) -> None:
    s3_client = boto3.client('s3', endpoint_url=S3_URL)
    # s3_client = boto3.client('s3', endpoint_url=s3_url, aws_access_key_id=s3_access_key, aws_secret_access_key=s3_secret_key)

    files = s3_client.list_objects(Bucket=S3_BUCKET)

    if "Contents" in files:
        for key in files['Contents']:
            key = key['Key']
            path = Path(key)
            print(key)

            loader_cls = LOADERS.get(path.suffix.lower())
            if loader_cls is None:
                print(f"Skipping '{key}' (unsupported extension '{path.suffix}').")
                continue

            s3_client.download_file(S3_BUCKET, key, path.name)

            loader = (
                loader_cls(path.name, encoding="utf-8") if loader_cls is TextLoader 
                else loader_cls(path.name)
            )

            doc = loader.load()

            ingest_document(doc, embeddings)
            print(f"Loaded {path.name} ({len(doc)} page(s))")
            os.remove(path.name)
    else:
        print("empty")
        
    print("end")

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

    ingest_documents(embeddings)

    print("Done.")
    return 0

#==============================================================================

if __name__ == "__main__":
    sys.exit(main())

#==============================================================================
