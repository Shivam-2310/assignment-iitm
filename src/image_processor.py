import base64
import os
from typing import Optional

# Try to import optional dependencies
try:
    from PIL import Image
    from io import BytesIO
    import pytesseract
    IMAGING_AVAILABLE = True
except ImportError:
    IMAGING_AVAILABLE = False

def process_image(base64_image: str) -> str:
    """
    Process a base64 encoded image using OCR to extract text.
    
    Args:
        base64_image: Base64 encoded image string
    
    Returns:
        Extracted text from the image, or a message if processing failed
    """
    # Check if imaging libraries are available
    if not 'IMAGING_AVAILABLE' in globals() or not IMAGING_AVAILABLE:
        print("Image processing is not available - PIL or pytesseract not installed")
        return "[Image processing not available in this deployment]"
        
    try:
        # Remove potential data URL prefix
        if ',' in base64_image:
            base64_image = base64_image.split(',')[1]
            
        # Decode base64 image
        image_data = base64.b64decode(base64_image)
        
        # Open image using PIL
        image = Image.open(BytesIO(image_data))
        
        # Use pytesseract to extract text
        text = pytesseract.image_to_string(image)
        
        return text.strip()
                    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return f"[Error processing image: {str(e)}]"

# Test function
def test_image_processing():
    """Test image processing with a sample base64 image."""
    # This is a very small dummy base64 encoded image for testing
    test_base64 = """
    iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAYAAACNiR0NAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6
    JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAABmJLR0QA/wD/AP+gvaeTAAAAB3RJTUUH
    5AQYEisxS9q2BgAAAWpJREFUOMvdlDFLw1AUhb+XNLZVqC1UcHBwENzrJoib4OBf6Q/o4OSou6O7m4tQ
    cKwgFYfSInbRwaGttrXJc0iqaZr0BS0IPXDhvXfPu5d7Lw/+OwKNllwDJTXGdS4XQKIxrk+AkurGdW6D
    OIl2AwWwpLpxna8gRmIqOlUhzJEE9YvE1mBK/eK6qSBdpoUk5w/V2z2dEct5BWrFPYdA1iGXASb0wrMz
    lqZvAW2UqK7L1qaZhjj0oq0r0cINyA7ILtAHBtE9pjYcYszLq8Ifs1SogGRQYgNYB04iXotSU4rDLfr9
    M/r9V1qtIUpNUWpKvz8AoNcb0OsNWKlVFhK9WzgXpTZBjlHy3rCKHLHRbHO6u4fW+YU/Uz8/9BZS02Xs
    eu/rXx1CxQ3IllL6QOvsjn1+F0yuGfDkQrkMOAYkl4u+7ytQC2XElYarGE9L89oDrfDwByuXH2AqKvlX
    9n/jA0XQaVjAuOEZAAAAAElFTkSuQmCC
    """
    
    result = process_image(test_base64)
    print(f"Test result: {result}")
    
if __name__ == "__main__":
    test_image_processing()
