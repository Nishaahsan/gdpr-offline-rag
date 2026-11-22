"""Step 2 - Offline GDPR RAG chatbot (Streamlit + Ollama + ChromaDB).

Run:  streamlit run app.py
"""
import html
import re
import time

import chromadb
import ollama
import streamlit as st

DB_DIR = "chroma_db"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"
DEFAULT_TOP_K = 5

EXAMPLES = [
    ("⏱️", "Breach notification", "What is the deadline to notify a personal data breach?"),
    ("🗑️", "Right to erasure", "When can a data subject ask for erasure of their data?"),
    ("✅", "Valid consent", "What are the conditions for valid consent?"),
    ("💶", "Fines", "What are the maximum administrative fines?"),
]

SYSTEM = (
    "You are a GDPR study assistant. Answer ONLY from the CONTEXT below. "
    "Cite every claim with the article label exactly as shown, e.g. [Art. 17]. "
    "If the context does not contain the answer, reply exactly: "
    "\"The provided GDPR text does not cover this.\" "
    "Keep the answer under 180 words, in simple English. "
    "This is educational information, not legal advice."
)

st.set_page_config(page_title="GDPR Assistant", page_icon="⚖️", layout="centered")

st.markdown(
    """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 820px;}
.hero {text-align: center; padding: 1.2rem 0 0.4rem 0;}
.hero h1 {font-size: 2.3rem; margin-bottom: 0.2rem;}
.hero p {color: #6b7280; font-size: 1.02rem; margin-top: 0;}
.badge {display: inline-block; background: #e8f1fb; color: #1f4e79; border-radius: 999px;
        padding: 2px 12px; font-size: 0.78rem; font-weight: 600; margin: 0 3px 8px 3px;}
.chip {display: inline-block; background: #eef2ff; color: #3730a3; border: 1px solid #c7d2fe;
       border-radius: 8px; padding: 2px 9px; font-size: 0.8rem; margin: 2px 4px 2px 0;}
div[data-testid="stButton"] > button {border-radius: 12px; text-align: left; height: 3.2rem;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_collection():
    return chromadb.PersistentClient(path=DB_DIR).get_collection("gdpr")


def index_size():
    try:
        return get_collection().count()
    except Exception:
        return None


def retrieve(question: str, k: int):
    col = get_collection()
    docs, metas, seen = [], [], set()

    # If the user typed "Article 17" directly, fetch that article first
    for num in re.findall(r"\b(?:article|art\.?)\s*(\d{1,3})\b", question, flags=re.I):
        got = col.get(where={"article": int(num)})
        for d, m, i in zip(got["documents"], got["metadatas"], got["ids"]):
            if i not in seen:
                seen.add(i); docs.append(d); metas.append(m)

    q_emb = ollama.embed(model=EMBED_MODEL, input=question)["embeddings"][0]
    res = col.query(query_embeddings=[q_emb], n_results=k)
    for d, m, i in zip(res["documents"][0], res["metadatas"][0], res["ids"][0]):
        if i not in seen:
            seen.add(i); docs.append(d); metas.append(m)
    return docs[: k + 2], metas[: k + 2]


def build_messages(question, docs, metas):
    context = "\n\n".join(f"[{m['label']}] {d}" for d, m in zip(docs, metas))
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"},
    ]


def show_extras(docs, metas, info):
    """Source chips, response time and the retrieved chunks under an answer."""
    if not docs:
        return
    labels = list(dict.fromkeys(m["label"] for m in metas))
    chips = " ".join(f'<span class="chip">{html.escape(l)}</span>' for l in labels)
    st.markdown(f"**Sources:** {chips}", unsafe_allow_html=True)
    if info:
        st.caption(info)
    with st.expander("View retrieved text"):
        for d, m in zip(docs, metas):
            st.markdown(f"**{m['label']}** - {m['title']}")
            st.write(d[:500] + ("..." if len(d) > 500 else ""))


def set_pending(q: str):
    st.session_state.pending = q


if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## ⚖️ GDPR Assistant")
    st.caption("Offline RAG chatbot")
    n = index_size()
    if n:
        st.success(f"Index ready: {n} chunks")
    else:
        st.error("No index found. Run `python ingest.py`.")
    top_k = st.slider("Passages to retrieve", 3, 8, DEFAULT_TOP_K,
                      help="How many GDPR passages are given to the model.")
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown("1. Your question is turned into an embedding\n"
                "2. The closest GDPR articles are retrieved\n"
                "3. A local LLM answers only from them, with citations")
    st.markdown("---")
    st.caption(f"LLM: `{LLM_MODEL}`  \nEmbeddings: `{EMBED_MODEL}`  \nRuns fully on your computer.")
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.caption("⚠️ Educational project. Not legal advice.")

# ------------------------------------------------------------------ welcome screen
if not st.session_state.messages:
    st.markdown(
        """
<div class="hero">
  <h1>⚖️ GDPR Assistant</h1>
  <p>Ask questions about the EU General Data Protection Regulation and get answers with article citations.</p>
  <span class="badge">100% offline</span><span class="badge">Cited answers</span><span class="badge">99 articles</span>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("##### Try an example")
    cols = st.columns(2)
    for i, (icon, title, q) in enumerate(EXAMPLES):
        with cols[i % 2]:
            st.button(f"{icon}  {title}", key=f"ex{i}", use_container_width=True,
                      on_click=set_pending, args=(q,), help=q)

# ------------------------------------------------------------------ chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🧑‍💻" if m["role"] == "user" else "⚖️"):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            show_extras(m.get("docs"), m.get("metas"), m.get("info"))

# ------------------------------------------------------------------ new question
question = st.chat_input("Ask about the GDPR... (e.g. What is the right to erasure?)")
if st.session_state.get("pending"):
    question = st.session_state.pop("pending")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(question)

    t0 = time.time()
    try:
        with st.spinner("Searching the GDPR text..."):
            docs, metas = retrieve(question, top_k)
    except Exception as e:
        st.error(f"Index not found or Ollama is not running. Run `python ingest.py` first. ({e})")
        st.stop()

    with st.chat_message("assistant", avatar="⚖️"):
        box, answer = st.empty(), ""
        stream = ollama.chat(model=LLM_MODEL, messages=build_messages(question, docs, metas),
                             stream=True, options={"temperature": 0})
        for part in stream:
            answer += part["message"]["content"]
            box.markdown(answer + "▌")
        box.markdown(answer)
        info = f"Answered in {time.time() - t0:.1f} s · {len(docs)} passages used"
        show_extras(docs, metas, info)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "docs": docs, "metas": metas, "info": info})
