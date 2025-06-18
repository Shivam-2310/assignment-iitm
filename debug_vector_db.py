print("Script started")
import os
try:
    print("Importing asyncio")
    import asyncio
    print("Importing dotenv")
    from dotenv import load_dotenv
    print("Loading env vars")
    load_dotenv()
    print("Env vars loaded")
    
    print("Creating data directory")
    os.makedirs('data', exist_ok=True)
    print("Data directory created or exists")

    print("Importing VectorDB")
    from src.vector_db import VectorDB
    print("VectorDB imported successfully")
    
    print("Initializing VectorDB")
    vector_db = VectorDB()
    print("VectorDB initialized successfully")
    
    # Create a tiny test document
    test_docs = [
        {
            "content": "This is a test document for the vector database.",
            "metadata": {
                "title": "Test Document",
                "url": "https://example.com/test",
            }
        }
    ]
    
    print("Adding test documents")
    vector_db.add_documents(test_docs, source_type="test")
    print("Test documents added")
    
    print("Testing similarity search")
    results = vector_db.similarity_search("test document", top_k=1)
    print(f"Search results: {results}")
    
    print("Saving vector database")
    vector_db.save('data/vector_db_test')
    print("Vector database saved successfully")
    
    print("Debug completed successfully")
    
except Exception as e:
    print(f"ERROR: {str(e)}")
    import traceback
    print(traceback.format_exc())
    
print("Script ended")
