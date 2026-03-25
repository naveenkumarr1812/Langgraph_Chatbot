import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from chatbot_backend import (
    chatbot,
    generate_thread_title,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)

st.set_page_config(page_title="QueryBot", page_icon="🤖", layout="wide")


# -----------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------
def generate_thread_id() -> str:
    return str(uuid.uuid4())


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    _add_thread(thread_id)
    st.session_state["message_history"] = []


def _add_thread(thread_id: str):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id: str) -> list:
    try:
        state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
        return state.values.get("messages", [])
    except Exception:
        return []


# -----------------------------------------------------------------------
# Session state initialisation
# -----------------------------------------------------------------------
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

# thread_titles maps thread_id -> short human-readable title
if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = {}

_add_thread(st.session_state["thread_id"])

thread_key: str = st.session_state["thread_id"]
thread_docs: dict = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads: list = st.session_state["chat_threads"][::-1]
selected_thread = None


# -----------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------
with st.sidebar:
    st.title("QueryBot")

    if st.button("➕ New Chat", use_container_width=True):
        reset_chat()
        st.rerun()

    st.divider()

    # Document status
    if thread_docs:
        latest_doc = list(thread_docs.values())[-1]
        st.success(
            f"📄 **{latest_doc.get('filename')}**\n\n"
            f"{latest_doc.get('chunks')} chunks · {latest_doc.get('documents')} page(s)"
        )
    else:
        st.info("No PDF uploaded yet for this chat.")

    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_pdf:
        if uploaded_pdf.name in thread_docs:
            st.info(f"`{uploaded_pdf.name}` is already indexed for this chat.")
        else:
            with st.status("Indexing PDF…", expanded=True) as status_box:
                try:
                    summary = ingest_pdf(
                        uploaded_pdf.getvalue(),
                        thread_id=thread_key,
                        filename=uploaded_pdf.name,
                    )
                    thread_docs[uploaded_pdf.name] = summary
                    status_box.update(label="✅ PDF indexed", state="complete", expanded=False)
                except RuntimeError as exc:
                    status_box.update(label="❌ Indexing failed", state="error", expanded=True)
                    st.error(f"**Error:** {exc}")
                except ValueError as exc:
                    status_box.update(label="❌ PDF error", state="error", expanded=True)
                    st.error(f"**PDF problem:** {exc}")
                except Exception as exc:
                    status_box.update(label="❌ Unexpected error", state="error", expanded=True)
                    st.error(f"**Unexpected error:** {exc}")

    st.divider()

    # Past conversations — show title instead of UUID
    st.subheader("💬 Past Conversations")
    if not threads:
        st.caption("No past conversations yet.")
    else:
        for tid in threads:
            title = st.session_state["thread_titles"].get(
                tid,
                f"Chat {str(tid)[:8]}…"   # fallback until title is generated
            )
            # Highlight the active thread
            is_active = tid == thread_key
            label = f"{'▶ ' if is_active else ''}{title}"
            if st.button(label, key=f"thread-{tid}", use_container_width=True):
                selected_thread = tid


# -----------------------------------------------------------------------
# Main chat area
# -----------------------------------------------------------------------
st.title("QueryBot")
st.caption("Ask questions about your PDF, use web search, or calculate anything.")

# Render existing messages
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask about your document, search the web, or calculate…")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate a title from the very first message of this thread
    if thread_key not in st.session_state["thread_titles"]:
        with st.spinner(""):
            title = generate_thread_title(user_input)
            st.session_state["thread_titles"][thread_key] = title

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_holder: dict = {"box": None}

        def ai_only_stream():
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}`…", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}`…",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage) and message_chunk.content:
                    yield message_chunk.content

        try:
            ai_message = st.write_stream(ai_only_stream())
        except Exception as exc:
            st.error(f"Error generating response: {exc}")
            ai_message = f"[Error: {exc}]"

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message or ""}
    )

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"📄 {doc_meta.get('filename')} — "
            f"{doc_meta.get('chunks')} chunks"
        )

st.divider()

# -----------------------------------------------------------------------
# Load a past conversation when sidebar button is clicked
# -----------------------------------------------------------------------
if selected_thread:
    st.session_state["thread_id"] = selected_thread
    messages = load_conversation(selected_thread)

    temp_messages = []
    for msg in messages:
        if isinstance(msg, (HumanMessage, AIMessage)) and msg.content:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": msg.content})

    st.session_state["message_history"] = temp_messages
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()