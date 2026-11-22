#!/usr/bin/env python3

import streamlit as st
import weaviate

from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

############################################################
# Config
############################################################

# Embeddings

emb_url     = "https://nv-embedqa-e5-v5.project-user-andrea-carboni.serving.ai-application.pcai0104.ld7.hpecolo.net/v1"
emb_model   = "nvidia/nv-embedqa-e5-v5"
emb_api_key = "eyJhbGciOiJSUzI1NiIsImtpZCI6ImNPcHh0SUNfanR1ZUpNSG5EdzNEU09Xd3ljZko0b2pQM1lXNUVGaGZ4VlUifQ.eyJhdWQiOlsiYXBpIiwiaXN0aW8tY2EiXSwiZXhwIjoxODA1OTg3MDM5LCJpYXQiOjE3ODAwNjcwMzksImlzcyI6Imh0dHBzOi8va3ViZXJuZXRlcy5kZWZhdWx0LnN2Yy5jbHVzdGVyLmxvY2FsIiwianRpIjoiYWY5N2NlNTUtN2Y3Zi00NWVlLWEyOWQtZDNlOGVhY2UyMTNjIiwia3ViZXJuZXRlcy5pbyI6eyJuYW1lc3BhY2UiOiJ1aSIsInNlcnZpY2VhY2NvdW50Ijp7Im5hbWUiOiJpc3ZjLWVwLTE3ODAwNjcwMzkzOTUiLCJ1aWQiOiI2NjIwNjUwYi1kNDRiLTRlZGUtYTdiOC0yYjhiYmNkZjgwMjMifX0sIm5iZiI6MTc4MDA2NzAzOSwic3ViIjoic3lzdGVtOnNlcnZpY2VhY2NvdW50OnVpOmlzdmMtZXAtMTc4MDA2NzAzOTM5NSJ9.faj7zqk2e6ARmanw2sj0tFAGZBmztVKj-inJ2QkURi8M9bKNRWWXx72psWHOWNJlnPcULo-L_wq0gejnH-pyLIJKCKiIPh5PBtchLkfI28uswxE8lCv0rrM2adVAMIB7p_3BJfYcRSNpburkYNrI_MXvkseHiPvK3aWhxknsbpM3H1dSiTY2jAAjzrij8G7CciD0cfZ7-u3K23KKzXuHTzKo1FhTEDlDuc8C2wDZ0P42zoO7wFIsQs8ipSQtbhWPwyf1VEUqrAvP1-T7ZiWCaGmBpIhcBjSDzWqUCG92cTQuQDFcW9J4TS352KulBb2pBRyvy07r2lkD9J0Hjxvepg"

# VectorStore

vst_http    = "weaviate.weaviate.svc.cluster.local"
vst_grpc    = "weaviate-grpc.weaviate.svc.cluster.local"
vst_index   = "Doc"

# LLM

llm_url     = "https://local-gpt-oss-120b.project-user-andrea-carboni.serving.ai-application.pcai0104.ld7.hpecolo.net/v1"
llm_model   = "/mnt/models"
llm_api_key = "eyJhbGciOiJSUzI1NiIsImtpZCI6ImNPcHh0SUNfanR1ZUpNSG5EdzNEU09Xd3ljZko0b2pQM1lXNUVGaGZ4VlUifQ.eyJhdWQiOlsiYXBpIiwiaXN0aW8tY2EiXSwiZXhwIjoxODA1OTg2NjI3LCJpYXQiOjE3ODAwNjY2MjcsImlzcyI6Imh0dHBzOi8va3ViZXJuZXRlcy5kZWZhdWx0LnN2Yy5jbHVzdGVyLmxvY2FsIiwianRpIjoiNjUwYWFjOGItNjRhZC00MDU1LWJmMTctODJkYWVmYTViNmRhIiwia3ViZXJuZXRlcy5pbyI6eyJuYW1lc3BhY2UiOiJ1aSIsInNlcnZpY2VhY2NvdW50Ijp7Im5hbWUiOiJpc3ZjLWVwLTE3ODAwNjY2Mjc3MDYiLCJ1aWQiOiI0ZGVlMjU3Yy1hZjI0LTQ4ZDItYmQxOS0yZmQwNDgyZTk3YjMifX0sIm5iZiI6MTc4MDA2NjYyNywic3ViIjoic3lzdGVtOnNlcnZpY2VhY2NvdW50OnVpOmlzdmMtZXAtMTc4MDA2NjYyNzcwNiJ9.Atn-BuE1WEcfxaKmRZTqkiKe9-SWTeKg3nubM4SIaWQbOzGCYMQpgrPVSd52dcBj99ydDUo-J9CTbfm0gNYc46BYIA9bWY_hYHUfBCQUnTHqQ6cVBLV29-3-Pmeo6ltMLb1dBqYPMf2OMn9jjUAcXIR-1rO1dJj3084dz1X6fz_aut8kDhm-W8KXCsma5aAEpbq5ivTzq5za2WkjsIHzpQVcUUfBRGRae7DCY75StzEn_lgmS6wimakoTm5cTbletP3UML6fJu_P5kw4oAHoR-7l-XJUeqx4jAI6lRxqMFmTO5J-IcG3xDkHnlRJjwqcLBtmKbdW1ewxfIr_G-_XgA"

############################################################
# Functions
############################################################


def title():
    st.write("# HPE Testing App - RAG Demo")


def get_chunks(query: str):
    # Get the relevant chunks from the vector store
    retriever = st.session_state.vector_store.as_retriever(k=5)
    docs = retriever.get_relevant_documents(query)
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
    client = weaviate.connect_to_custom(
        http_host=vst_http,  # Hostname for the HTTP API connection
        http_port=80,        # Default is 80, WCD uses 443
        http_secure=False,   # Whether to use https (secure) for the HTTP API connection
        grpc_host=vst_grpc,  # Hostname for the gRPC API connection
        grpc_port=50051,     # Default is 50051, WCD uses 443
        grpc_secure=False,   # Whether to use a secure channel for the gRPC API connection
        skip_init_checks=True
    )

    try:
        if client.is_ready():
            # You can now interact with your Weaviate instance
            print("Successfully connected to Weaviate with custom configuration!")
            embedder = NVIDIAEmbeddings(
                model=emb_model,
                base_url=emb_url,
                api_key=emb_api_key,
                truncate="END",
            )
            return WeaviateVectorStore(index_name=vst_index, client=client, embedding=embedder, text_key="text")
        else:
            print("Failed to connect to Weaviate.")
    except Exception as e:
        print(f"Error connecting to Weaviate: {e}")


def create_prompt():

    template = """Your job is to use patient
    reviews to answer questions about their experience at a hospital.
    Use the following context to answer questions. Be as detailed
    as possible, but don't make up any information that's not
    from the context. If you don't know an answer, say you don't know.

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
    st.session_state.model = ChatNVIDIA(base_url=llm_url, model=llm_model, api_key=llm_api_key)

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

