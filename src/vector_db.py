import os
import json
import pickle
import sys
import re
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Use scikit-learn instead of sentence-transformers
try:
    print("Importing sklearn...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    print("sklearn imported successfully")
except ImportError as e:
    print(f"Error importing sklearn: {e}")
    print("Please make sure scikit-learn is installed:")
    print("pip install scikit-learn")
    sys.exit(1)

class VectorDB:
    def __init__(self):
        # Storage for documents and their embeddings
        self.documents = []  # List to store document content
        self.embeddings = []  # List to store vector embeddings
        self.metadata = []  # List to store metadata (source URL, type, etc.)
        
        # Numpy array for embeddings (will be initialized when adding documents)
        self.embeddings_array = None
        
        # Maximum context size for chunking documents
        self.max_chunk_size = 1000  # Characters (approximate)
        
        # Semantic search parameters
        self.top_k = 5  # Number of results to return
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(max_features=384, stop_words='english')
        self.fitted = False
        
        print("VectorDB initialized with TF-IDF vectorizer")
        
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text using TF-IDF vectorizer."""
        try:
            # Clean text - remove special chars, lowercase
            text = re.sub(r'[^\w\s]', ' ', text.lower())
            
            # For a single text, we need a list
            if not self.fitted:
                # First time, fit and transform
                vector = self.vectorizer.fit_transform([text])
                self.fitted = True
            else:
                # Subsequently, just transform
                vector = self.vectorizer.transform([text])
                
            # Convert sparse matrix to dense array and get the first (only) row
            dense_vector = vector.toarray()[0]
            
            # Normalize to unit length
            norm = np.linalg.norm(dense_vector)
            if norm > 0:
                dense_vector = dense_vector / norm
                
            return dense_vector.tolist()
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Fallback to returning a zero embedding
            return [0.0] * 384
    
    def _chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into chunks suitable for embedding."""
        chunks = []
        
        # Simple implementation: split by paragraphs and combine into chunks
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        
        current_chunk = ""
        for paragraph in paragraphs:
            # Rough estimation of tokens (1 token ≈ 4 chars)
            if len(current_chunk) + len(paragraph) < self.max_chunk_size * 4:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    # Create a copy of metadata for this chunk
                    chunk_metadata = metadata.copy()
                    chunks.append({
                        "text": current_chunk.strip(),
                        "metadata": chunk_metadata
                    })
                current_chunk = paragraph + "\n\n"
        
        # Add the last chunk if not empty
        if current_chunk:
            chunk_metadata = metadata.copy()
            chunks.append({
                "text": current_chunk.strip(),
                "metadata": chunk_metadata
            })
            
        return chunks
    
    def add_documents(self, documents: List[Dict[str, Any]], source_type: str):
        """Add documents to the vector database."""
        all_texts = []
        temp_chunks = []
        
        for doc in documents:
            # Each document should have "content" and "metadata"
            content = doc["content"]
            metadata = doc.get("metadata", {})
            
            # Add source type to metadata
            metadata["source_type"] = source_type
            
            # Chunk the document
            chunks = self._chunk_text(content, metadata)
            
            # Add all chunks to temporary storage
            for chunk in chunks:
                all_texts.append(chunk["text"])
                temp_chunks.append(chunk)
                
        try:
            # Fit vectorizer on all document texts
            all_texts = [doc['text'] for doc in chunked_docs]
            self.vectorizer.fit(all_texts)
            self.fitted = True
            
            # Get embeddings for all texts
            for i, doc in enumerate(chunked_docs):
                embedding = self._get_embedding(doc['text'])
                self.embeddings.append(embedding)
                self.documents.append(doc['text'])
                self.metadata.append(doc['metadata'])
            
            # Convert embeddings to numpy array
            self._build_embeddings_array()
            
        except Exception as e:
            print(f"Error fitting vectorizer or building index: {e}")
    
    def _build_embeddings_array(self):
        """Convert embeddings list to numpy array for similarity search."""
        if not self.embeddings:
            return
        
        try:
            # Convert embeddings to numpy array
            self.embeddings_array = np.array(self.embeddings).astype('float32')
            
            # Get the dimension of the embeddings
            dimension = self.embeddings_array.shape[1]
            
            print(f"Built embeddings array with {len(self.embeddings)} vectors of dimension {dimension}")
        except Exception as e:
            print(f"Error building embeddings array: {e}")
    
    def similarity_search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search for documents similar to the query using cosine similarity."""
        try:
            if not self.documents or self.embeddings_array is None:
                print("No documents or embeddings available for search")
                return []
            
            if top_k is None:
                top_k = self.top_k
            
            # Get query embedding
            query_embedding = np.array([self._get_embedding(query)])
            
            # Use cosine similarity for search
            similarities = cosine_similarity(query_embedding, self.embeddings_array)[0]
            
            # Get indices of top-k most similar documents
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                # Only include results with positive similarity
                if similarities[idx] <= 0:
                    continue
                    
                results.append({
                    "content": self.documents[idx],
                    "metadata": self.metadata[idx],
                    "score": float(similarities[idx])  # Convert to float for JSON serialization
                })
            
            print(f"Found {len(results)} similar documents")
            return results
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []
    
    def save(self, directory: str):
        """Save the vector database to disk."""
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save documents
        with open(os.path.join(directory, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        
        # Save embeddings
        with open(os.path.join(directory, "embeddings.pkl"), "wb") as f:
            pickle.dump(self.embeddings, f)
        
        # Save metadata
        with open(os.path.join(directory, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)
        
        # Save embeddings array if it exists
        if self.embeddings_array is not None:
            with open(os.path.join(directory, "embeddings_array.pkl"), "wb") as f:
                pickle.dump(self.embeddings_array, f)
    
    def load(self, directory: str):
        """Load the vector database from disk."""
        # Load documents
        with open(os.path.join(directory, "documents.pkl"), "rb") as f:
            self.documents = pickle.load(f)
        
        # Load embeddings
        with open(os.path.join(directory, "embeddings.pkl"), "rb") as f:
            self.embeddings = pickle.load(f)
        
        # Load metadata
        with open(os.path.join(directory, "metadata.pkl"), "rb") as f:
            self.metadata = pickle.load(f)
        
        # Load embeddings array if it exists
        embeddings_array_path = os.path.join(directory, "embeddings_array.pkl")
        if os.path.exists(embeddings_array_path):
            with open(embeddings_array_path, "rb") as f:
                self.embeddings_array = pickle.load(f)
        else:
            # If no saved array, but we have embeddings, build the array
            if self.embeddings:
                self._build_embeddings_array()
