import chromadb
import os
import uuid


class VectorStoreManager:

    def __init__(
            self,
            persist_directory="vector_store",
            collection_name="pdf_documents"):

        self.collection_name = collection_name
        self.persist_directory = persist_directory

        self.collection = None
        self.client = None

        self._initialize_store()

    def _initialize_store(self):

        os.makedirs(
            self.persist_directory,
            exist_ok=True
        )

        self.client = chromadb.PersistentClient(
            path=self.persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

    def add_documents(
            self,
            documents,
            embeddings,
            batch_size=1000):

        if len(documents) != len(embeddings):
            raise ValueError(
                "Number of documents and embeddings do not match."
            )

        ids = []
        metadata_list = []
        documents_content = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(
                zip(documents, embeddings)):

            ids.append(
                f"doc_{uuid.uuid4()}"
            )

            metadata = dict(doc.metadata)

            metadata["doc_index"] = i
            metadata["content_length"] = len(
                doc.page_content
            )

            metadata_list.append(metadata)

            documents_content.append(
                doc.page_content
            )

            embeddings_list.append(
                embedding.tolist()
            )

        total_docs = len(ids)

        for start in range(
                0,
                total_docs,
                batch_size):

            end = start + batch_size

            self.collection.add(
                ids=ids[start:end],
                metadatas=metadata_list[start:end],
                documents=documents_content[start:end],
                embeddings=embeddings_list[start:end]
            )

            print(
                f"Added documents {start} to {min(end,total_docs)}"
            )

        print(
            f"Successfully stored {total_docs} chunks."
        )