from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import base64
import os
from typing import Optional, List, Dict, Any
import json
from dotenv import load_dotenv
import time
from contextlib import asynccontextmanager

# Try to import asyncio/aiohttp, but provide fallbacks if not available
try:
    import asyncio
except ImportError:
    # Create a minimal asyncio mock for basic functionality
    class AsyncioMock:
        @staticmethod
        async def create_task(coro):
            return await coro
            
        @staticmethod
        async def wait_for(coro, timeout):
            return await coro
            
    asyncio = AsyncioMock()

# Import our modules
from src.scraper import init_scrapers, scrape_discourse, scrape_course_content
from src.vector_db import VectorDB
from src.image_processor import process_image
from src.qa_system import generate_answer

# Load environment variables
load_dotenv()

# Initialize vector database
vector_db = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize vector database on startup
    global vector_db
    
    # Check if we already have data
    vector_db_exists = os.path.exists('data/vector_db')
    
    # Initialize vector database
    vector_db = VectorDB()
    
    if not vector_db_exists:
        print("First time setup: Scraping data and creating vector database...")
        # Initialize scrapers
        init_scrapers()
        
        # Scrape data
        course_content = await scrape_course_content()
        discourse_posts = await scrape_discourse()
        
        # Process and add to vector database
        vector_db.add_documents(course_content, source_type="course_content")
        vector_db.add_documents(discourse_posts, source_type="discourse")
        
        # Save vector database
        vector_db.save('data/vector_db')
        print("Vector database created and saved.")
    else:
        print("Loading existing vector database...")
        vector_db.load('data/vector_db')
        print("Vector database loaded.")
    
    yield
    
    # Cleanup on shutdown
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

class QuestionRequest(BaseModel):
    question: str
    image: Optional[str] = None  # Base64 encoded image

class LinkInfo(BaseModel):
    url: str
    text: str

class AnswerResponse(BaseModel):
    answer: str
    links: List[LinkInfo]

@app.post("/api/", response_model=AnswerResponse)
async def answer_question(request: QuestionRequest):
    start_time = time.time()
    
    # Set a timeout for the entire process (30 seconds as per requirement)
    timeout = 30.0
    remaining_time = timeout - (time.time() - start_time)
    
    if remaining_time <= 0:
        raise HTTPException(status_code=408, detail="Request timeout")
    
    try:
        # Process question
        question = request.question
        
        # Process image if provided
        image_text = None
        if request.image:
            try:
                image_text = process_image(request.image)
                question = f"{question}\n\nImage content: {image_text}"
            except Exception as e:
                print(f"Error processing image: {e}")
                # Continue without image text if there's an error
        
        # Get answer with timeout
        answer_task = asyncio.create_task(
            generate_answer(vector_db, question, remaining_time - 1)  # Leave 1 second buffer
        )
        
        try:
            answer, links = await asyncio.wait_for(answer_task, timeout=remaining_time - 0.5)
        except asyncio.TimeoutError:
            # If we're about to timeout, return a fallback response
            return AnswerResponse(
                answer="I apologize, but I couldn't generate a complete answer within the time limit. Please try asking a more specific question.",
                links=[]
            )
        
        # Format and return response
        link_objects = [LinkInfo(url=link["url"], text=link["text"]) for link in links]
        
        return AnswerResponse(
            answer=answer,
            links=link_objects
        )
        
    except Exception as e:
        remaining_time = timeout - (time.time() - start_time)
        if remaining_time <= 0:
            raise HTTPException(status_code=408, detail="Request timeout")
        else:
            raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
