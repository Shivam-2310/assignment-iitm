print("Starting test")
try:
    print("Importing SentenceTransformer")
    from sentence_transformers import SentenceTransformer
    print("SentenceTransformer imported")
    
    print("Loading model")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded successfully")
    
    print("Testing model with a simple sentence")
    result = model.encode("This is a test sentence")
    print(f"Encoding successful, shape: {result.shape}")
    
    print("All tests passed!")
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    print(traceback.format_exc())
