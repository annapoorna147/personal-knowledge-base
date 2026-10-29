
import hashlib

import streamlit as st

from src.document_processor import extract_text, split_into_chunks
from src.vector_store import VectorStore
from src.answer_generator import generate_answer


st.set_page_config(
    page_title="Personal Knowledge Base AI",
    page_icon="🧠",
    layout="wide",
)


@st.cache_resource
def get_vector_store():
    return VectorStore()


if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


st.title("🧠 Personal Knowledge Base AI")
st.caption("Your documents. Your knowledge. One searchable workspace.")

with st.sidebar:
    st.header("Knowledge Library")
    st.write("Upload PDFs or TXT files to build your library.")
    st.caption("Document vectors are stored locally.")

    if st.button("Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()


st.subheader("1. Upload your documents")

uploaded_files = st.file_uploader(
    "Choose PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.write(f"Selected: {len(uploaded_files)} document(s)")

    if st.button("Process and index documents", type="primary"):
        try:
            with st.spinner(
                "Loading the embedding model and indexing documents..."
            ):
                store = get_vector_store()
                total_chunks = 0
                skipped_files = 0

                for uploaded_file in uploaded_files:
                    file_bytes = uploaded_file.getvalue()

                    signature = hashlib.sha256(
                        uploaded_file.name.encode("utf-8")
                        + file_bytes
                    ).hexdigest()

                    if signature in st.session_state.indexed_files:
                        skipped_files += 1
                        continue

                    pages = extract_text(uploaded_file)
                    document_chunks = []

                    for page in pages:
                        chunks = split_into_chunks(page["text"])

                        for chunk in chunks:
                            document_chunks.append({
                                "text": chunk,
                                "source": page["source"],
                                "page": page["page"],
                            })

                    if document_chunks:
                        added = store.add_chunks(document_chunks)
                        total_chunks += added
                        st.session_state.indexed_files.add(signature)
                    else:
                        st.warning(
                            f"No extractable text found in "
                            f"{uploaded_file.name}."
                        )

            st.success(
                f"Indexing complete! Added {total_chunks} text chunks."
            )

            if skipped_files:
                st.info(
                    f"Skipped {skipped_files} document(s) "
                    "already indexed in this session."
                )

        except Exception as exc:
            st.error(f"Indexing failed: {exc}")


st.divider()
st.subheader("2. Ask your knowledge base")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("View retrieved sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"**{source['source']}** — "
                        f"Page {source['page']}"
                    )
                    st.caption(
                        source["text"][:700]
                        + ("..." if len(source["text"]) > 700 else "")
                    )

question = st.chat_input(
    "Ask a question about your documents..."
)

if question:
    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.spinner("Searching documents and generating an answer..."):
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

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": matches,
        })

    except Exception as exc:
        st.error(f"Unable to answer your question: {exc}")


