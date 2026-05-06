import os
import re
import base64
import tempfile
import chromadb
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from chromadb.utils import embedding_functions
from ingest import ingest
from database import init_db, log_query, get_total_queries, get_avg_confidence, get_most_queried_documents, get_recent_queries, delete_all_logs

load_dotenv()
client = OpenAI()
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)
chroma = chromadb.PersistentClient(path="./chroma_db")
init_db()

st.set_page_config(page_title="Visual RAG", page_icon="🔍", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .main-header p { color: #b8d4f0; margin: 0.3rem 0 0; font-size: 0.95rem; }
    .doc-count {
        background: #1e3a5f;
        color: #b8d4f0;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        font-size: 0.85rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    .welcome-box {
        background: #1a2744;
        border: 1px solid #2d6a9f;
        border-radius: 12px;
        padding: 2.5rem;
        text-align: center;
        margin-top: 3rem;
    }
    .welcome-box h2 { color: #b8d4f0; font-size: 1.4rem; }
    .welcome-box p { color: #7a9bbf; font-size: 0.95rem; }
    .source-badge {
        background: #1e3a5f;
        color: #b8d4f0;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.3rem;
    }
    .doc-item {
        background: #1a2744;
        border: 1px solid #2d4a6f;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        color: #b8d4f0;
    }
    .confidence-high { color: #4caf50; font-weight: 600; }
    .confidence-mid { color: #ff9800; font-weight: 600; }
    .confidence-low { color: #f44336; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# Helper to get doc collections
def get_doc_collections():
    try:
        collections = chroma.list_collections()
        return [c for c in collections if c.name.startswith("doc_")]
    except:
        return []

# Helper to get confidence label
def confidence_label(relevance):
    if relevance >= 70:
        return f'<span class="confidence-high">🟢 {relevance}% match</span>'
    elif relevance >= 40:
        return f'<span class="confidence-mid">🟡 {relevance}% match</span>'
    else:
        return f'<span class="confidence-low">🔴 {relevance}% match</span>'

# Sidebar
with st.sidebar:
    st.markdown("## 🔍 Visual RAG")
    st.markdown("---")

    doc_collections = get_doc_collections()
    doc_count = len(doc_collections)
    st.markdown(f'<div class="doc-count">📚 {doc_count} document{"s" if doc_count != 1 else ""} indexed</div>', unsafe_allow_html=True)

    st.markdown("### Upload a PDF")
    uploaded_file = st.file_uploader("", type=["pdf"])
    if uploaded_file is not None:
        if st.button("⬆️ Ingest Document", use_container_width=True):
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                doc_name = uploaded_file.name.replace(".pdf", "")
                doc_name = re.sub(r'[^a-zA-Z0-9._-]', '_', doc_name)
                if doc_name[0].isdigit():
                    doc_name = "doc" + doc_name
                ingest(tmp_path, custom_name=doc_name)
                os.unlink(tmp_path)
            st.success(f"✅ {uploaded_file.name} ingested!")
            st.rerun()

    st.markdown("---")

    # Document list with chunk count and delete
    if doc_count > 0:
        st.markdown("### Indexed documents")
        for col in doc_collections:
            doc_display_name = col.name.replace("doc_", "")
            try:
                chunk_count = col.count()
            except:
                chunk_count = 0
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f'<div class="doc-item">📄 {doc_display_name}<br><small>{chunk_count} chunks</small></div>', unsafe_allow_html=True)
            with col2:
                if st.button("🗑️", key=f"del_{doc_display_name}", help=f"Delete {doc_display_name}"):
                    try:
                        chroma.delete_collection(col.name)
                        master = chroma.get_or_create_collection("all_documents", embedding_function=openai_ef)
                        existing = master.get(where={"source": doc_display_name})
                        if existing["ids"]:
                            master.delete(ids=existing["ids"])
                        st.success(f"Deleted {doc_display_name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting: {e}")

    st.markdown("---")
    st.markdown("### Search in")

    if doc_count == 0:
        st.info("Upload a PDF to get started.")
        selected_doc = "All documents"
    else:
        doc_names = [c.name.replace("doc_", "") for c in doc_collections]
        options = ["All documents"] + doc_names
        selected_doc = st.selectbox("", options)

    st.markdown("---")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main area header
st.markdown("""
<div class="main-header">
    <h1>🔍 Visual RAG — Technical Document Q&A</h1>
    <p>Ask questions about your ingested documents. The AI finds relevant text and images to answer.</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["💬 Chat", "📊 Analytics"])

# ── Analytics tab ──
with tab2:
    st.markdown("### Query Analytics")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Queries", get_total_queries())
    with col2:
        st.metric("Avg Confidence", f"{get_avg_confidence()}%")
    with col3:
        st.metric("Documents Indexed", doc_count)

    st.markdown("---")

    st.markdown("#### Most Queried Documents")
    most_queried = get_most_queried_documents()
    if most_queried:
        for doc, count in most_queried:
            st.markdown(f"**{doc}** — {count} quer{'y' if count == 1 else 'ies'}")
    else:
        st.caption("No queries yet.")

    st.markdown("---")

    st.markdown("#### Recent Queries")
    recent = get_recent_queries()
    if recent:
        for timestamp, question, doc, confidence in recent:
            with st.expander(f"🕐 {timestamp} — {question[:60]}..."):
                st.markdown(f"**Document:** {doc}")
                st.markdown(f"**Confidence:** {confidence}%")
    else:
        st.caption("No queries yet.")

    st.markdown("---")
    if st.button("🗑️ Clear all logs", use_container_width=True):
        delete_all_logs()
        st.success("All logs cleared!")
        st.rerun()

# ── Chat tab ──
with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Welcome screen
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-box">
            <h2>👋 Welcome to Visual RAG</h2>
            <p>Upload a PDF in the sidebar and start asking questions.<br>
            The AI will search through text <strong>and</strong> images to find your answer.</p>
            <br>
            <p>💡 Try asking things like:<br>
            <em>"What is the main conclusion?"</em> &nbsp;|&nbsp;
            <em>"Explain the diagram on page 3"</em> &nbsp;|&nbsp;
            <em>"What are the key findings?"</em></p>
        </div>
        """, unsafe_allow_html=True)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "images" in message and message["images"]:
                for img in message["images"]:
                    img_bytes = base64.b64decode(img["img_b64"])
                    st.image(img_bytes, caption=f"📄 Page {img['page']} — {img['source']}", width=400)

    # Chat input
    if question := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            search_msg = f"Searching across {doc_count} document{'s' if doc_count != 1 else ''}..." if selected_doc == "All documents" else f"Searching in {selected_doc}..."
            with st.spinner(search_msg):

                if selected_doc == "All documents":
                    collection = chroma.get_or_create_collection("all_documents", embedding_function=openai_ef)
                else:
                    collection = chroma.get_or_create_collection(f"doc_{selected_doc}", embedding_function=openai_ef)

                results = collection.query(query_texts=[question], n_results=3)
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]

                context = ""
                images_found = []
                sources_used = []
                for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
                    context += f"\n[Source {i+1} - Page {meta['page']} - Type: {meta['type']} - Document: {meta['source']}]\n{doc}\n"
                    if meta["type"] == "image":
                        images_found.append(meta)
                    relevance = round((1 - dist) * 100, 1)
                    sources_used.append({
                        "name": meta["source"],
                        "page": meta["page"],
                        "relevance": relevance
                    })

                user_content = [{"type": "text", "text": f"Using the following context, answer this question: {question}\n\nContext:\n{context}"}]
                for img_meta in images_found:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_meta['img_b64']}"}
                    })

                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant answering questions about technical documents. Always cite which document and page your answer comes from."},
                        {"role": "user", "content": user_content}
                    ],
                    max_tokens=500
                )

                answer = response.choices[0].message.content
                st.markdown(answer)

                # Source highlights with confidence
                st.markdown("---")
                st.markdown("**📎 Sources used:**")
                for src in sources_used:
                    st.markdown(
                        f'<span class="source-badge">{src["name"]} — page {src["page"]}</span> {confidence_label(src["relevance"])}',
                        unsafe_allow_html=True
                    )

                if images_found:
                    st.markdown("**🖼️ Referenced images:**")
                    for img in images_found:
                        img_bytes = base64.b64decode(img["img_b64"])
                        st.image(img_bytes, caption=f"Page {img['page']} — {img['source']}", width=400)

                avg_confidence = round(sum([s["relevance"] for s in sources_used]) / len(sources_used), 1) if sources_used else 0
                log_query(
                    question=question,
                    answer=answer,
                    document_searched=selected_doc,
                    avg_confidence_score=avg_confidence,
                    num_images_retrieved=len(images_found),
                    num_chunks_retrieved=len(documents)
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "images": images_found
                })