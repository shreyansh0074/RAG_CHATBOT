import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend import (
    chatbot,
    ingest_pdf,
    thread_document_metadata,
)


# =========================== Utilities ===========================
def generate_thread_id():
    return str(uuid.uuid4())


def rebuild_history(thread_id: str):
    """Reconstruct display-friendly message_history from SQLite for a thread."""
    history = []
    for msg in chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    ).values.get("messages", []):
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and msg.content:
            history.append({"role": "assistant", "content": msg.content})
    return history


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


# ======================= Session Initialization ===================
if "thread_id" not in st.session_state:
    # Reuse the thread_id from the URL (?thread=...) if this is a page refresh
    # of an existing conversation; otherwise start a fresh one. Only this
    # visitor's own browser ever has this ID in its address bar, so it's not
    # shared with anyone else.
    url_thread = st.query_params.get("thread")
    if url_thread:
        st.session_state["thread_id"] = url_thread
        st.session_state["message_history"] = rebuild_history(url_thread)
    else:
        st.session_state["thread_id"] = generate_thread_id()
        st.session_state["message_history"] = []

if "chat_threads" not in st.session_state:
    # Only track threads created in THIS browser session — never seed from the
    # shared backend, which would otherwise leak every visitor's past chats
    # into every other visitor's sidebar.
    st.session_state["chat_threads"] = []

# Reconcile the URL on every single run (not just when a thread is first
# created) so the address bar always matches the active thread, even right
# after "New Chat" — this is what makes a manual refresh reliable.
st.query_params["thread"] = str(st.session_state["thread_id"])

add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
# Document metadata is read straight from the backend (which persists to disk),
# rather than kept only in session state, so it survives thread switches and restarts.
threads = st.session_state["chat_threads"][::-1]
selected_thread = None

# ============================ Sidebar ============================
st.sidebar.title("LangGraph PDF Chatbot")
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

current_doc_meta = thread_document_metadata(thread_key)
if current_doc_meta:
    st.sidebar.success(
        f"Using `{current_doc_meta.get('filename')}` "
        f"({current_doc_meta.get('chunks')} chunks from {current_doc_meta.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if current_doc_meta.get("filename") == uploaded_pdf.name:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            try:
                summary = ingest_pdf(
                    uploaded_pdf.getvalue(),
                    thread_id=thread_key,
                    filename=uploaded_pdf.name,
                )
                status_box.update(label="✅ PDF indexed", state="complete", expanded=False)
                st.rerun()
            except ValueError as e:
                status_box.update(label="❌ Indexing failed", state="error", expanded=True)
                st.sidebar.error(str(e))
            except Exception:
                status_box.update(label="❌ Indexing failed", state="error", expanded=True)
                st.sidebar.error("Something went wrong while indexing this PDF. Please try again.")

st.sidebar.subheader("Past conversations")
if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for thread_id in threads:
        if st.sidebar.button(str(thread_id), key=f"side-thread-{thread_id}"):
            selected_thread = thread_id

# ============================ Main Layout ========================
st.title("Multi Utility Chatbot")

# Chat area
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask about your document or use tools")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            try:
                for message_chunk, _ in chatbot.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode="messages",
                ):
                    if isinstance(message_chunk, ToolMessage):
                        tool_name = getattr(message_chunk, "name", "tool")
                        if status_holder["box"] is None:
                            status_holder["box"] = st.status(
                                f"🔧 Using `{tool_name}` …", expanded=True
                            )
                        else:
                            status_holder["box"].update(
                                label=f"🔧 Using `{tool_name}` …",
                                state="running",
                                expanded=True,
                            )

                    if isinstance(message_chunk, AIMessage) and message_chunk.content:
                        yield message_chunk.content
            except Exception:
                yield "\n\n Something went wrong generating a response. Please try again."

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )

st.divider()

if selected_thread:
    st.session_state["thread_id"] = selected_thread
    messages = load_conversation(selected_thread)

    temp_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            temp_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and msg.content:
            # Skip AIMessages that only carry tool_calls with no visible content.
            temp_messages.append({"role": "assistant", "content": msg.content})
    st.session_state["message_history"] = temp_messages
    st.rerun()