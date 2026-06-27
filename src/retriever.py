class RAGRetriever:

    def __init__(
            self,
            embedding_manager,
            vector_store):

        self.embedding_manager = embedding_manager
        self.vector_store = vector_store

    def retrieve(
            self,
            query,
            top_k=3):

        query_embedding = self.embedding_manager.generate_embeddings(
            [query]
        )[0]

        results = self.vector_store.collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=top_k
        )

        return results["documents"][0]