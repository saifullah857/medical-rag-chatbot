from src.loader import load_all_pdfs
from src.splitter import split_docs
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager

documents = load_all_pdfs()

chunks = split_docs(documents)

embedding_manager = EmbeddingManager()

texts = [
    doc.page_content
    for doc in chunks
]

embeddings = embedding_manager.generate_embeddings(
    texts
)

vector_store = VectorStoreManager()

vector_store.add_documents(
    chunks,
    embeddings
)

print("Vector Store Created Successfully")