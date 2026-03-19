import chromadb # pyre-ignore[21]
import uuid
import os

class VectorMemory:
    def __init__(self, db_path="./chroma_db", collection_name="episodic_memory"):
        self.client = chromadb.PersistentClient(path=db_path)
        # Using default embedding function for sentence transformation
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_memory(self, text, metadata=None):
        """
        Stores a semantic memory using the vector database.
        """
        memory_id = str(uuid.uuid4())
        meta = metadata if metadata else {"type": "conversation"}
        self.collection.add(
            documents=[text],
            metadatas=[meta],
            ids=[memory_id]
        )
        return memory_id

    def retrieve_memories(self, query_text, n_results=3, similarity_threshold=0.8):
        """
        Retrieves top memories that match the query text.
        In Chroma, distances for the default embedding are generally L2 or Cosine.
        We'll retrieve N results and return them if they are close enough.
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )

        memories = []
        if not results['documents'] or not results['documents'][0]:
            return memories

        docs = results['documents'][0]
        distances = results['distances'][0]

        for doc, distance in zip(docs, distances):
            # Chroma distance: lower is better (usually L2 distance for default model).
            # We use distance to roughly threshold "vividness".
            # For simplicity, we just return the top N matching documents.
            # You can tune this distance logic based on the specific embedding model used.
            if distance < 1.5:  # This threshold is empirical for L2 distance.
                memories.append(doc)
        
        return memories
