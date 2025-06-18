# IIT Madras TDS Virtual Teaching Assistant

A virtual teaching assistant API for the Tools in Data Science course at IIT Madras. This API automatically answers student questions based on course content and Discourse posts.

## Features

- **Automated Responses**: Answers student questions using LLM technology powered by GPT-3.5-turbo
- **Context-Aware**: Retrieves relevant information from course materials and discussion forums
- **Image Processing**: Extracts text from screenshots to provide context-aware answers
- **Fast Response Time**: Guaranteed response within 30 seconds
- **Source Attribution**: Includes links to relevant sources for further reading

## Tech Stack

- **FastAPI**: Modern web framework for building APIs
- **OpenAI API**: Used for embeddings and answer generation
- **Vector Database**: Custom implementation for semantic search
- **Tesseract OCR**: For extracting text from images
- **Docker**: For containerization and deployment

## Setup Instructions

### Prerequisites

- Python 3.8+
- OpenAI API key
- Tesseract OCR installed on your system (for local development)

### Local Development

1. Clone this repository:
   ```
   git clone <repository-url>
   cd abhedaya_assignment
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Install Tesseract OCR:
   - **Windows**: Download and install from [here](https://github.com/UB-Mannheim/tesseract/wiki)
   - **Linux**: `sudo apt-get install tesseract-ocr`
   - **macOS**: `brew install tesseract`

5. Create a `.env` file based on `.env.example` and add your API keys:
   ```
   cp .env.example .env
   # Edit .env with your API keys
   ```

6. Run the application:
   ```
   uvicorn main:app --reload
   ```

7. The API will be available at `http://localhost:8000/api/`

### Docker Deployment

1. Build the Docker image:
   ```
   docker build -t tds-virtual-ta .
   ```

2. Run the container:
   ```
   docker run -p 8000:8000 --env-file .env -d tds-virtual-ta
   ```

3. The API will be available at `http://localhost:8000/api/`

### Cloud Deployment

This application can be deployed on various cloud platforms:

#### Option 1: Render.com
1. Create a new web service on Render
2. Connect to your GitHub repository
3. Set environment variables from your `.env` file
4. Deploy with Docker

#### Option 2: Fly.io
1. Install the Fly CLI: `curl -L https://fly.io/install.sh | sh`
2. Log in: `flyctl auth login`
3. Create app: `flyctl apps create`
4. Deploy: `flyctl deploy`

## API Usage

### Endpoint: `/api/`

**Method**: POST

**Request Format**:
```json
{
  "question": "Your question here",
  "image": "Optional base64 encoded image"
}
```

**Response Format**:
```json
{
  "answer": "Detailed answer to the question",
  "links": [
    {
      "url": "https://example.com/source",
      "text": "Source description"
    }
  ]
}
```

### Example Usage

```bash
curl "https://your-deployed-api.com/api/" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"Should I use gpt-4o-mini which AI proxy supports, or gpt3.5 turbo?\", \"image\": \"$(base64 -w0 project-tds-virtual-ta-q1.webp)\"}"
```

## Evaluation

To evaluate the application:

1. Update `project-tds-virtual-ta-promptfoo.yaml` with your API URL
2. Run evaluation:
   ```
   npx -y promptfoo eval --config project-tds-virtual-ta-promptfoo.yaml
   ```

## Project Structure

```
.
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration
├── README.md               # This file
├── .env.example            # Environment variables template
├── data/                   # Directory for vector database storage
└── src/
    ├── __init__.py
    ├── vector_db.py        # Vector database implementation
    ├── scraper.py          # Web scraping functionality
    ├── image_processor.py  # OCR and image processing
    └── qa_system.py        # Question answering logic
```

## Notes

- For the first run, the system will scrape data and build the vector database, which might take some time
- Subsequent runs will use the cached database for faster startup
- The model used for answer generation is gpt-3.5-turbo-0125 as specified in the course requirements

## License

[MIT License](LICENSE)
