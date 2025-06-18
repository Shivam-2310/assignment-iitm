import os
import base64
import sys
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import pytesseract
from src.image_processor import process_image

def create_test_image(output_path="test_image.png"):
    """Create a test image with text for OCR testing"""
    # Create a white image
    img = Image.new('RGB', (800, 400), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    
    # Try to use a system font
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        # Fall back to default if Arial not available
        try:
            font = ImageFont.load_default()
        except:
            print("Could not load any font, using simple text")
            font = None
    
    # Add text to the image
    test_text = """
    TDS Jan 2025 - Important Requirements
    
    For Question 8 in GA5:
    - You must use gpt-3.5-turbo-0125 model specifically
    - Even if AI Proxy supports gpt-4o-mini, use OpenAI API directly
    - Calculate token usage using the tokenizer from the lecture
    - Multiply by the given rate to determine cost
    
    Submission Deadline: April 30, 2025, 11:59 PM IST
    """
    
    if font:
        d.text((50, 50), test_text, fill=(0, 0, 0), font=font)
    else:
        d.text((50, 50), test_text, fill=(0, 0, 0))
    
    # Save the image
    img.save(output_path)
    print(f"Test image created at {output_path}")
    return output_path

def test_ocr(image_path="test_image.png"):
    """Test OCR on the created image"""
    if not os.path.exists(image_path):
        print(f"Image file {image_path} not found. Creating a test image...")
        image_path = create_test_image(image_path)
    
    print(f"Testing OCR on {image_path}...")
    
    # Read the image
    with open(image_path, "rb") as image_file:
        # Convert the image to base64
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    
    # Process the image using our module
    extracted_text = process_image(encoded_image)
    
    print("\nExtracted Text:")
    print("=" * 50)
    print(extracted_text)
    print("=" * 50)
    
    return extracted_text

if __name__ == "__main__":
    # Check if Tesseract is installed
    try:
        pytesseract.get_tesseract_version()
        print("Tesseract is installed.")
    except Exception as e:
        print(f"Error: Tesseract is not installed or not in PATH. {e}")
        print("Please install Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki")
        sys.exit(1)
    
    # Create a test image if it doesn't exist
    image_path = "test_image.png"
    if not os.path.exists(image_path):
        image_path = create_test_image(image_path)
    
    # Test OCR
    test_ocr(image_path)
