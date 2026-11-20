import os
import httpx
from dotenv import load_dotenv
from crewai import Agent, Crew, LLM, Task, Process
from crewai_tools import MCPServerAdapter
from phoenix.otel import register

#==============================================================================

load_dotenv()
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

LLM_MODEL    = os.getenv("LLM_MODEL")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY  = os.getenv("LLM_API_KEY")
SSL_VERIFY   = os.getenv("SSL_VERIFY").lower() not in {"false", "0", "no"}

MCP_URL_DATABASE= os.getenv("MCP_URL_DATABASE")
MCP_URL_VECTOR  = os.getenv("MCP_URL_VECTOR")

PHOENIX_ENDPOINT = os.getenv("PHOENIX_ENDPOINT")
PROJECT          = "agentic-rag"

tracer_provider = register( project_name=PROJECT, 
                            endpoint=f"{PHOENIX_ENDPOINT}/v1/traces", 
                            auto_instrument=True)

#==============================================================================

def refresh_token() -> str:
    with open("/etc/secrets/ezua/.auth_token", "r") as file:
        token = file.read().strip()
    return token

#==============================================================================

def patch_httpx_ssl() -> None:
    for client_cls in (httpx.Client, httpx.AsyncClient):
        original_init = client_cls.__init__

        def patched_init(client, *args, _original_init=original_init, **kwargs):
            kwargs.setdefault("verify", False)
            _original_init(client, *args, **kwargs)

        client_cls.__init__ = patched_init

#==============================================================================

def build_crew(tools) -> Crew:
    analyst = Agent(
        role="Financial Analyst",
        goal=(
            "Answer financial questions accurately by using the available MCP tools to query live data. "
            "When calculating a compensation for a cancellation, increase the compensation "
            "by the percentage associated to the passenger's card (silver,gold or platinum). "
            "Assume the distance is in kilometers."
        ),
        backstory=(
            "An experienced financial analyst connected to the company's "
            "financial data platform through MCP tools. You call the right "
            "tools with correct parameters and reason carefully over the "
            "returned data before answering."
        ),
        tools=list(tools),
        llm=LLM(model=LLM_MODEL, base_url=LLM_BASE_URL, api_key=LLM_API_KEY),
        verbose=False,
    )
    
    #-----------------------------------------

    task = Task(
        description=(
            "Answer the user's question:\n{question}\n\n"
            "Use the available tools to fetch the data you need. If a tool "
            "call fails, adjust the parameters or try a related tool before "
            "giving up. Ground every claim in the data the tools return."
            "When needing to specify a catalog, use 'agenticrag'"
        ),
        expected_output=(
            "A concise answer to the question, citing the concrete values "
            "returned by the tools."
        ),
        agent=analyst,
    )

    #-----------------------------------------

    return Crew(agents      = [analyst], 
                tasks       = [task], 
                planning    = True, 
                planning_llm= LLM(model=LLM_MODEL, base_url=LLM_BASE_URL, api_key=LLM_API_KEY),
                process     = Process.sequential,
                verbose     = False)

#==============================================================================

def main() -> None:
    if not SSL_VERIFY:
        patch_httpx_ssl()

    database_params = {
        "url"      : MCP_URL_DATABASE,
        "transport": "streamable-http",
        "headers"  : {"Authorization": f"Bearer {refresh_token()}"},
    }

    vector_params = {
        "url"      : MCP_URL_VECTOR,
        "transport": "streamable-http",
    }

    with MCPServerAdapter(database_params, connect_timeout=120) as dbMcp:
        with MCPServerAdapter(vector_params, connect_timeout=120) as vecMcp:
            print(f"Building crew...")
            crew = build_crew([*dbMcp, *vecMcp])

            question = "I'm Alice. Show me my cancellation refund."
            print(f"Executing task: {question}")
            result = crew.kickoff(inputs={"question": question})
            print(result.raw)

#==============================================================================

if __name__ == "__main__":
    main()

#==============================================================================
