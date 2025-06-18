import requests
import json
import base64
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API endpoint
API_URL = "http://localhost:8000/api/"

def test_text_question():
    """Test the API with a text-only question"""
    question = "Should I use gpt-4o-mini which AI proxy supports, or gpt3.5 turbo?"
    
    payload = {
        "question": question
    }
    
    print(f"Sending question: {question}")
    start_time = time.time()
    
    response = requests.post(API_URL, json=payload)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Response received in {elapsed_time:.2f} seconds")
    
    if response.status_code == 200:
        result = response.json()
        print("\nAnswer:")
        print(result["answer"])
        print("\nLinks:")
        for link in result["links"]:
            print(f"- {link['text']}: {link['url']}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

def test_image_question():
    """Test the API with a question + image"""
    # Path to an image file
    image_path = "project-tds-virtual-ta-q1.webp"
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Warning: Image file '{image_path}' not found. Skipping image test.")
        return
    
    # Read and encode the image
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    
    question = "What does this screenshot show about model requirements?"
    
    payload = {
        "question": question,
        "image": encoded_image
    }
    
    print(f"\nSending question with image: {question}")
    start_time = time.time()
    
    response = requests.post(API_URL, json=payload)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Response received in {elapsed_time:.2f} seconds")
    
    if response.status_code == 200:
        result = response.json()
        print("\nAnswer:")
        print(result["answer"])
        print("\nLinks:")
        for link in result["links"]:
            print(f"- {link['text']}: {link['url']}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    print("Testing Virtual TA API...")
    test_text_question()
    test_image_question()
    print("\nTests completed.")
