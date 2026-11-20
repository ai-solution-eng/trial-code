# rag-search-mcp

MCP server exposing the **search** half of a RAG pipeline: embeds a query with NVIDIA NIM
embeddings (`nvidia/nv-embedqa-e5-v5`) and retrieves the top chunks from a Qdrant collection
via LangChain.

## Tools

| Tool | Description |
|---|---|
| `search_knowledge_base(query: str)` | Returns top-k chunks as JSON: `{query, results: [{content, score, metadata}]}`. Score is cosine similarity (higher is better). |

## Setup

```bash
uv sync
cp .env.example .env   # then fill in your values
```

Configuration (`.env` or environment):

| Variable | Required | Default | Description |
|---|---|---|---|
| `NVIDIA_API_KEY` | yes | — | NVIDIA NIM API key (`nvapi-...`) |
| `NVIDIA_BASE_URL` | no | `https://integrate.api.nvidia.com/v1` | Point to a self-hosted NIM if needed |
| `EMBEDDING_MODEL` | no | `nvidia/nv-embedqa-e5-v5` | 1024-dim embedding model |
| `QDRANT_URL` | yes | — | e.g. `http://localhost:6333` |
| `QDRANT_API_KEY` | no | — | For Qdrant Cloud |
| `QDRANT_COLLECTION` | yes | — | Collection to search |
| `SEARCH_TOP_K` | no | `5` | Number of chunks returned |

## Prerequisite

Ingestion is out of scope; the Qdrant collection must exist beforehand with vectors of
dimension **1024** (nv-embedqa-e5-v5). Documents were assumed to be embedded with the same
model (passage input type).

```bash
curl -X PUT $QDRANT_URL/collections/knowledge_base \
  -H 'Content-Type: application/json' \
  -d '{"vectors": {"size": 1024, "distance": "Cosine"}}'
```

## Run

Streamable HTTP transport; the MCP endpoint is `http://localhost:8000/mcp`.

```bash
uv run server.py
```

Server settings can be overridden via env, e.g. `FASTMCP_HOST`, `FASTMCP_PORT`.

### Docker

Config is passed via environment, not the excluded `.env`:

```bash
docker build -t rag-search-mcp .
docker run --rm -p 8000:8000 --env-file .env rag-search-mcp
```

Example client config:

```json
{
  "mcpServers": {
    "rag-search": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

Dev inspection: `uv run mcp dev server.py`
