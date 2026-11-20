import argparse
import os
import re
import sys
from pathlib import Path
from collections.abc import Iterable, Sequence
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

#==============================================================================

load_dotenv()

EMBED_URL   = os.environ.get("EMBED_URL",   "")
EMBED_KEY   = os.environ.get("EMBED_KEY",   "")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")

QDRANT_URL        = os.environ.get("QDRANT_URL",        "http://qdrant.qdrant.svc.cluster.local")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "documents")
TOP_K             = int(os.environ.get("TOP_K",         "5"))

LLM_URL   = os.environ.get("LLM_URL",   "")
LLM_KEY   = os.environ.get("LLM_KEY",   "")
LLM_MODEL = os.environ.get("LLM_MODEL", "")


SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant. Answer the user's question using "
    "ONLY the information in the provided context. If the context does not "
    "contain the answer, reply that you don't know. Do not use outside knowledge."
)

#==============================================================================

def format_docs(docs: Sequence[Document]) -> str:
    parts = []
    for doc in docs:
        source = Path(doc.metadata.get("source", "unknown")).name
        page = doc.metadata.get("page")
        origin = f"{source} p.{page + 1}" if page is not None else source
        parts.append(f"[{origin}]\n{doc.page_content}")
    return "\n\n".join(parts)

#==============================================================================

def build_chain(retriever: Runnable, llm: ChatNVIDIA) -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", f"{SYSTEM_PROMPT}\n\nContext:\n{{context}}"),
            ("human", "{question}"),
        ]
    )
    answer_chain: Runnable = (
        RunnablePassthrough.assign(
            context=lambda inputs: format_docs(inputs["context"])
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=answer_chain)

#==============================================================================

def print_sources(docs: Iterable[Document]) -> None:
    seen = set()
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        label = f"  - {Path(source).name}"
        if page is not None:
            label += f" (page {page + 1})"
        print(label)

#==============================================================================

def main() -> int:
    client     = QdrantClient(url=QDRANT_URL, prefer_grpc=False, timeout=30)
    embeddings = NVIDIAEmbeddings(model=EMBED_MODEL, base_url=EMBED_URL, api_key=EMBED_KEY, truncate="END")
    store      = QdrantVectorStore(client=client, collection_name=QDRANT_COLLECTION, embedding=embeddings)
    retriever  = store.as_retriever(search_kwargs={"k": TOP_K})
    llm        = ChatOpenAI(base_url=LLM_URL, api_key=LLM_KEY, model=LLM_MODEL)
    chain      = build_chain(retriever, llm)

    question = input("\nQuestion> ").strip()
    result = chain.invoke(question)
    print(f"\n{result['answer']}")
    print("\nSources:")
    print_sources(result["context"])
    return 0

#==============================================================================

if __name__ == "__main__":
    sys.exit(main())

#==============================================================================
#   llm = ChatNVIDIA(base_url=LLM_URL, api_key=LLM_KEY, model=LLM_MODEL, temperature=0, max_tokens=1024)

# Sample question: tell me about refund policies about flight cancellations