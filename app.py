import streamlit as st

from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager
from src.retriever import RAGRetriever
from src.llm import llm
from src.rag_chain import generate_output

# ----------------------------
# PAGE CONFIG
# ---------------------Z-------
st.set_page_config(
    page_title="Medical RAG Chatbot",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Medical RAG-Based Assistant")

st.caption("Ask medical questions based on your knowledge base")

# ----------------------------
# SESSION STATE (Chat Memory)
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# LOAD COMPONENTS (CACHE HEAVY OBJECTS)
# ----------------------------
@st.cache_resource
def load_embedding_manager():
    return EmbeddingManager()

@st.cache_resource
def load_vector_store():
    return VectorStoreManager()

@st.cache_resource
def load_retriever(_embedding_manager, _vector_store):
    return RAGRetriever(_embedding_manager, _vector_store)

embedding_manager = load_embedding_manager()
vector_store = load_vector_store()
retriever = load_retriever(embedding_manager, vector_store)

# ----------------------------
# DISPLAY CHAT HISTORY
# ----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ----------------------------
# USER INPUT
# ----------------------------
if prompt := st.chat_input("Ask a medical question..."):

    # user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.write(prompt)

    # ----------------------------
    # ASSISTANT RESPONSE WITH SPINNER
    # ----------------------------
    with st.chat_message("assistant"):
        with st.spinner("🧠 Analyzing medical knowledge..."):

            try:
                answer = generate_output(
                    prompt,
                    retriever,
                    llm
                )

            except Exception as e:
                answer = f"❌ Error: {str(e)}"

        st.write(answer)

    # save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })