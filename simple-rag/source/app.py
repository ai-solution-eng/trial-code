#!/usr/bin/env python3

import os

import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_openai import ChatOpenAI
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

############################################################
# Config
############################################################

load_dotenv()

EMBED_URL   = os.environ.get("EMBED_URL",   "")
EMBED_KEY   = os.environ.get("EMBED_KEY",   "")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")

QDRANT_URL        = os.environ.get("QDRANT_URL",        "http://qdrant.qdrant.svc.cluster.local")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "anywhere")

LLM_URL   = os.environ.get("LLM_URL",   "")
LLM_KEY   = os.environ.get("LLM_KEY",   "")
LLM_MODEL = os.environ.get("LLM_MODEL", "")

############################################################
# Functions
############################################################


def title():
    st.write("# HPE Testing App - RAG Demo")


def get_chunks(query: str):
    # Get the relevant chunks from the vector store
    retriever = st.session_state.vector_store.as_retriever(k=5)
    docs = retriever.invoke(query)
    return [doc.page_content for doc in docs]


def generate_ai_response(query: str):
    # Query the AI.
    prompt = create_prompt()
    chain = (
        {"context": st.session_state.vector_store.as_retriever(k=5), "question": RunnablePassthrough()}
        | prompt
        | st.session_state.model
        | StrOutputParser()
    )
    resp = chain.invoke(query)

    return resp


def create_vector_store():
    client     = QdrantClient(url=QDRANT_URL, prefer_grpc=False, timeout=30)
    embeddings = NVIDIAEmbeddings(model=EMBED_MODEL, base_url=EMBED_URL, api_key=EMBED_KEY, truncate="END")
    store      = QdrantVectorStore(client=client, collection_name=QDRANT_COLLECTION, embedding=embeddings)

    return store


def create_prompt():

    template = """You are a retrieval-augmented assistant. Answer the user's question using
    ONLY the information in the provided context. If the context does not contain the answer, 
    reply that you don't know. Do not use outside knowledge.
    {context}
    """

    system_prompt = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["context"], template=template,
        )
    )

    human_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["question"], template="{question}",
        )
    )

    messages = [system_prompt, human_prompt]

    return ChatPromptTemplate(input_variables=["context", "question"], messages=messages)


############################################################
# States
############################################################


if "messages" not in st.session_state:
    # Each item: {"query": str, "response": str, "chunks": List[str]}
    st.session_state.messages = []

if "query" not in st.session_state:
    st.session_state.query = ""

if "vector_store" not in st.session_state:
    st.session_state.vector_store = create_vector_store()

if "model" not in st.session_state:
    #st.session_state.model = ChatNVIDIA(base_url=llm_url, model=llm_model, api_key=llm_api_key)
    st.session_state.model = ChatOpenAI(base_url=LLM_URL, api_key=LLM_KEY, model=LLM_MODEL)


############################################################
# Page layout
############################################################

title()

left, right = st.columns([2, 1])

with left:
    st.subheader("Ask a question")
    query = st.text_area("Your query",
                         key="query",
                         height=120,
                         label_visibility="collapsed",
                         placeholder="Type your question here...")

    sent = st.button("Send", type="primary", use_container_width=True)
    if sent and query.strip():
        chunks = get_chunks(query)
        response = generate_ai_response(query)
        st.session_state.messages.append({
            "query": query.strip(),
            "response": response,
            "chunks": chunks
        })
        st.write("RAGed response:", response)

with right:
    st.subheader("Relevant Chunks")
    if st.session_state.messages:
        for i, ch in enumerate(st.session_state.messages[-1]["chunks"], start=1):
            st.markdown(f"**{i}.** {ch}")
    else:
        st.caption("Chunks will appear here after you send a query.")

