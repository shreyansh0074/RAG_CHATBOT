from __future__ import annotations
import os
import sqlite3
import tempfile
import logging
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Annotated, Any, Dict, Optional, TypedDict
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ragB")

# -------------------
# 0. Config
# -------------------
INDEX_ROOT = Path("faiss_indexes")  # one subfolder per thread, persisted to disk
INDEX_ROOT.mkdir(exist_ok=True)
MAX_PDF_BYTES = 25 * 1024 * 1024  # 25 MB safety cap

# -------------------
# 1. LLM + embeddings
# -------------------
llm = ChatGroq(model="llama-3.3-70b-versatile")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -------------------
# 2. PDF retriever store (per thread)
# -------------------
# In-memory cache so we don't hit disk on every single tool call within a session.
# The actual source of truth is the on-disk FAISS index, so a server restart
# doesn't lose ingested documents (only the previous behavior did).
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _index_dir(thread_id: str) -> Path:
    return INDEX_ROOT / str(thread_id)


def _load_metadata(thread_id: str) -> Optional[dict]:
    meta_path = _index_dir(thread_id) / "meta.json"
    if not meta_path.exists():
        return None
    import json
    try:
        return json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return None


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread, checking memory then disk."""
    if not thread_id:
        return None
    thread_id = str(thread_id)

    if thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]

    index_path = _index_dir(thread_id)
    if index_path.exists():
        try:
            vector_store = FAISS.load_local(
                str(index_path), embeddings, allow_dangerous_deserialization=True
            )
            retriever = vector_store.as_retriever(
                search_type="similarity", search_kwargs={"k": 4}
            )
            _THREAD_RETRIEVERS[thread_id] = retriever
            meta = _load_metadata(thread_id)
            if meta:
                _THREAD_METADATA[thread_id] = meta
            return retriever
        except Exception:
            logger.exception("Failed to load FAISS index for thread %s", thread_id)
            return None

    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF, persist it to disk, and
    cache it in memory for the thread. Returns a summary dict for the UI.
    Raises ValueError for bad input so the caller can show a clean error.
    """
    if not file_bytes:
        raise ValueError("No file content received.")
    if len(file_bytes) > MAX_PDF_BYTES:
        raise ValueError(
            f"File is too large ({len(file_bytes) / 1e6:.1f} MB). "
            f"Limit is {MAX_PDF_BYTES / 1e6:.0f} MB."
        )

    thread_id = str(thread_id)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        if not docs:
            raise ValueError("Could not extract any text from this PDF (it may be scanned/image-only).")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        if not chunks:
            raise ValueError("PDF produced no usable text chunks.")

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        # Persist to disk so restarts don't lose the index.
        index_path = _index_dir(thread_id)
        index_path.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(index_path))

        summary = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
        import json
        (index_path / "meta.json").write_text(json.dumps(summary))

        _THREAD_RETRIEVERS[thread_id] = retriever
        _THREAD_METADATA[thread_id] = summary
        return summary

    except Exception:
        logger.exception("PDF ingestion failed for thread %s", thread_id)
        raise
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass


# -------------------
# 3. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def rag_tool(query: str, config: RunnableConfig) -> dict:
    """
    Retrieve relevant information from the PDF uploaded for this chat thread.
    Use this whenever the user asks about the uploaded document.
    """
    # thread_id is injected from the run config, not supplied by the model,
    # so the LLM can never get it wrong or omit it.
    thread_id = (config.get("configurable") or {}).get("thread_id")
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Ask the user to upload a PDF first.",
            "query": query,
        }

    try:
        result = retriever.invoke(query)
    except Exception:
        logger.exception("Retrieval failed for thread %s", thread_id)
        return {"error": "Retrieval failed unexpectedly.", "query": query}

    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


tools = [search_tool, rag_tool]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 5. Nodes
# -------------------
def chat_node(state: ChatState, config=None):
    """LLM node that may answer or request a tool call."""
    system_message = SystemMessage(
        content=(
            "You are a helpful assistant with two tools available: "
            "`rag_tool` for answering questions about the PDF the user has uploaded "
            "to this chat, and `search_tool` for general web search. "
            "Use `rag_tool` whenever the question could relate to the uploaded document. "
            "If `rag_tool` reports no document is indexed, ask the user to upload a PDF. "
            "Use `search_tool` for anything requiring current or general web information."
        )
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}


tool_node = ToolNode(tools)

# -------------------
# 6. Checkpointer
# -------------------
# check_same_thread=False is required for SqliteSaver's internal usage pattern.
# For a multi-user production deployment, prefer a per-request connection or a
# proper database-backed checkpointer over a single shared sqlite3 connection.
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# -------------------
# 7. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 8. Helpers
# -------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    thread_id = str(thread_id)
    if thread_id in _THREAD_RETRIEVERS:
        return True
    return _index_dir(thread_id).exists()


def thread_document_metadata(thread_id: str) -> dict:
    """Metadata for a thread's ingested PDF, checked in-memory then on disk."""
    thread_id = str(thread_id)
    if thread_id in _THREAD_METADATA:
        return _THREAD_METADATA[thread_id]
    meta = _load_metadata(thread_id)
    if meta:
        _THREAD_METADATA[thread_id] = meta
        return meta
    return {}
