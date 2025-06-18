import os
import json
import pickle
from typing import List, Dict, Any, Optional
import numpy as np
import time
import sys
from collections import Counter
import re
import hashlib

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

try:
    print("Importing faiss...")
    import faiss
    print("faiss imported successfully")
except ImportError as e:
    print(f"Error importing faiss: {e}")
    print("Please make sure faiss-cpu is installed:")
    print("pip install faiss-cpu")
    sys.exit(1)

class VectorDB:
    def __init__(self):
        # Storage for documents and their embeddings
        self.documents = []  # List to store document content
        self.embeddings = []  # List to store vector embeddings
        self.metadata = []  # List to store metadata (source URL, type, etc.)
        
        # FAISS index for fast similarity search (will be initialized when adding documents)
        self.index = None
        
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
            # Fit vectorizer on all texts at once if not already fitted
            if not self.fitted and all_texts:
                print(f"Fitting vectorizer on {len(all_texts)} documents")
                self.vectorizer.fit(all_texts)
                self.fitted = True
                print("Vectorizer fitted successfully")
            
            # Now process each chunk
            for chunk in temp_chunks:
                try:
                    # Get embedding for the chunk
                    embedding = self._get_embedding(chunk["text"])
                    
                    # Store document, embedding, and metadata
                    self.documents.append(chunk["text"])
                    self.embeddings.append(embedding)
                    self.metadata.append(chunk["metadata"])
                except Exception as e:
                    print(f"Error embedding chunk: {e}")
                    # Continue with other chunks
            
            # Build FAISS index with the embeddings
            self._build_faiss_index()
            
        except Exception as e:
            print(f"Error fitting vectorizer or building index: {e}")
    
    def _build_faiss_index(self):
        """Build a FAISS index for fast similarity search."""
        if not self.embeddings:
            return
        
        try:
            # Convert embeddings to numpy array
            embeddings_np = np.array(self.embeddings).astype('float32')
            
            # Get the dimension of the embeddings
            dimension = embeddings_np.shape[1]
            
            # Create a new index - using L2 distance
            self.index = faiss.IndexFlatL2(dimension)
            
            # Add the vectors to the index
            self.index.add(embeddings_np)
            
            print(f"Built FAISS index with {len(self.embeddings)} vectors of dimension {dimension}")
        except Exception as e:
            print(f"Error building FAISS index: {e}")
    
    def similarity_search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search for documents similar to the query."""
        try:
            if not self.documents or self.index is None:
                print("No documents or index available for search")
                return []
            
            if top_k is None:
                top_k = self.top_k
            
            # Get query embedding
            query_embedding = np.array([self._get_embedding(query)]).astype('float32')
            
            # Use FAISS for fast similarity search
            distances, indices = self.index.search(query_embedding, top_k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                # Skip if idx is -1 (means not enough results found)
                if idx == -1:
                    continue
                    
                results.append({
                    "content": self.documents[idx],
                    "metadata": self.metadata[idx],
                    "score": 1.0 - distances[0][i] / 100.0  # Convert distance to a similarity score
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
        
        # Save FAISS index if it exists
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(directory, "faiss_index.bin"))
    
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
        
        # Load FAISS index if it exists
        faiss_index_path = os.path.join(directory, "faiss_index.bin")
        if os.path.exists(faiss_index_path):
            self.index = faiss.read_index(faiss_index_path)
        else:
            # If no saved index, but we have embeddings, build the index
            if self.embeddings:
                self._build_faiss_index()
