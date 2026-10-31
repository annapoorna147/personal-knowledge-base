
import hashlib

import streamlit as st

from src.document_processor import extract_text, split_into_chunks
from src.vector_store import VectorStore
from src.answer_generator import generate_answer


st.set_page_config(
    page_title="Personal Knowledge Base AI",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Personal Knowledge Base AI")
st.caption(
    "Upload PDFs or TXT files and ask questions about their contents."
)


@st.cache_resource
def get_vector_store():
    """Create and reuse the vector store."""
    return VectorStore()


# Initialize session state.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()


# Sidebar: document upload and indexing.
with st.sidebar:
    st.header("Knowledge Library")
    st.write("Upload PDFs or TXT files to build your library.")

    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.write(f"Selected: {len(uploaded_files)} document(s)")

        if st.button("Process and index documents", type="primary"):
            store = get_vector_store()

            for uploaded_file in uploaded_files:
                file_bytes = uploaded_file.getvalue()
                file_hash = hashlib.sha256(file_bytes).hexdigest()

                file_signature = (
                    uploaded_file.name,
                    file_hash,
                )

                if file_signature in st.session_state.indexed_files:
                    st.info(
                        f"{uploaded_file.name} is already indexed "
                        "in this session."
                    )
                    continue

                try:
                    with st.spinner(
                        f"Processing {uploaded_file.name}..."
                    ):
                        # Returns a list of page dictionaries.
                        extracted_pages = extract_text(uploaded_file)

                        chunks_to_add = []

                        # Split each page's text separately.
                        for page in extracted_pages:
                            page_chunks = split_into_chunks(
                                page["text"]
                            )

                            for chunk_text in page_chunks:
                                chunks_to_add.append(
                                    {
                                        "text": chunk_text,
                                        "source": page["source"],
                                        "page": page["page"],
                                    }
                                )

                        if not chunks_to_add:
                            st.warning(
                                f"No extractable text was found in "
                                f"{uploaded_file.name}. "
                                "If this is a scanned PDF, OCR may be needed."
                            )
                            continue

                        added_count = store.add_chunks(chunks_to_add)

                        st.session_state.indexed_files.add(
                            file_signature
                        )

                    st.success(
                        f"Indexed {uploaded_file.name}: "
                        f"{added_count} text chunks added."
                    )

                except Exception as exc:
                    st.error(
                        f"Could not process {uploaded_file.name}: {exc}"
                    )

    st.divider()

    if st.button("Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()

    st.caption("Document vectors are stored locally.")


# Display previous chat messages.
st.subheader("Ask your knowledge base")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("View retrieved sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"**{source['source']}** — "
                        f"Page {source['page']}"
                    )
                    st.caption(source["text"][:700])


# Question input.
question = st.chat_input(
    "Ask a question about your documents..."
)

if question:
    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.spinner(
            "Searching documents and generating an answer..."
        ):
            store = get_vector_store()
            matches = store.search(question, top_k=4)

            answer = generate_answer(question, matches)

        with st.chat_message("assistant"):
            st.markdown(answer)

            if matches:
                with st.expander("View retrieved sources"):
                    for source in matches:
                        st.markdown(
                            f"**{source['source']}** — "
                            f"Page {source['page']}"
                        )
                        st.caption(source["text"][:700])

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": matches,
            }
        )

    except Exception as exc:
        st.error(f"An error occurred: {exc}")
