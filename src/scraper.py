import os
import json
import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any
import aiohttp
import requests
from bs4 import BeautifulSoup

# Configuration
COURSE_CONTENT_URL = "https://onlinedegree.iitm.ac.in/course_content/TDS_Jan_2025/"  # Example URL structure
DISCOURSE_BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
DISCOURSE_API_URL = f"{DISCOURSE_BASE_URL}/c/tools-in-data-science/70.json"  # Example category ID for TDS

# Authentication credentials (would need to be provided via env variables)
AUTH_USERNAME = os.getenv("AUTH_USERNAME")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD")
DISCOURSE_API_KEY = os.getenv("DISCOURSE_API_KEY")
DISCOURSE_API_USERNAME = os.getenv("DISCOURSE_API_USERNAME")

# Session objects
course_session = None
discourse_session = None

def init_scrapers():
    """Initialize scraper sessions with authentication."""
    global course_session, discourse_session
    
    # Initialize session for course content (may need different auth mechanism)
    course_session = requests.Session()
    if AUTH_USERNAME and AUTH_PASSWORD:
        # Example authentication - adjust based on actual authentication mechanism
        login_url = "https://onlinedegree.iitm.ac.in/login"
        login_data = {
            "username": AUTH_USERNAME,
            "password": AUTH_PASSWORD
        }
        course_session.post(login_url, data=login_data)
    
    # Initialize session for discourse
    discourse_session = requests.Session()
    if DISCOURSE_API_KEY:
        discourse_session.headers.update({
            "Api-Key": DISCOURSE_API_KEY,
            "Api-Username": DISCOURSE_API_USERNAME or "system"
        })

async def fetch_url_async(url: str, session_type: str = "course") -> str:
    """Fetch URL content asynchronously."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    if session_type == "discourse" and DISCOURSE_API_KEY:
        headers.update({
            "Api-Key": DISCOURSE_API_KEY,
            "Api-Username": DISCOURSE_API_USERNAME or "system"
        })
    
    async with aiohttp.ClientSession() as session:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with session.get(url, headers=headers, timeout=30) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        print(f"Error fetching {url}, status code: {response.status}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        else:
                            return ""
            except Exception as e:
                print(f"Exception fetching {url}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return ""
        return ""

def parse_course_content(html: str, url: str) -> Dict[str, Any]:
    """Parse course content HTML."""
    soup = BeautifulSoup(html, "html.parser")
    
    # Extract content from the page - adjust selectors based on actual page structure
    title_element = soup.select_one("h1.course-title")
    content_element = soup.select_one("div.course-content")
    
    title = title_element.text.strip() if title_element else "Unknown Title"
    content = content_element.text.strip() if content_element else ""
    
    # Extract links for reference
    links = []
    for link in soup.find_all("a", href=True):
        link_text = link.text.strip()
        link_url = link["href"]
        if link_url.startswith("/"):
            link_url = f"https://onlinedegree.iitm.ac.in{link_url}"
        links.append({"text": link_text, "url": link_url})
    
    return {
        "content": content,
        "metadata": {
            "title": title,
            "url": url,
            "links": links,
            "date": "2025-04-15",  # As per requirement, course content as of 15 Apr 2025
            "type": "course_content"
        }
    }

def parse_discourse_post(post_json: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a Discourse post from JSON."""
    topic_id = post_json.get("id")
    title = post_json.get("title", "")
    
    # Combine posts content
    posts_content = []
    for post in post_json.get("post_stream", {}).get("posts", []):
        username = post.get("username", "unknown")
        post_content = post.get("cooked", "")  # HTML content
        
        # Remove HTML tags to get plain text
        soup = BeautifulSoup(post_content, "html.parser")
        text_content = soup.get_text(separator="\n", strip=True)
        
        # Format post with username
        formatted_post = f"User: {username}\n{text_content}\n\n"
        posts_content.append(formatted_post)
    
    # Combine all posts into one content string
    full_content = f"Topic: {title}\n\n" + "".join(posts_content)
    
    # Extract links from post
    links = []
    topic_url = f"{DISCOURSE_BASE_URL}/t/{post_json.get('slug', '')}/{topic_id}"
    links.append({"text": title, "url": topic_url})
    
    # Get post date
    created_at = post_json.get("created_at")
    if created_at:
        try:
            date = datetime.fromisoformat(created_at.replace("Z", "+00:00")).strftime("%Y-%m-%d")
        except:
            date = "Unknown"
    else:
        date = "Unknown"
    
    return {
        "content": full_content,
        "metadata": {
            "title": title,
            "url": topic_url,
            "topic_id": topic_id,
            "links": links,
            "date": date,
            "type": "discourse_post"
        }
    }

async def scrape_course_content() -> List[Dict[str, Any]]:
    """Scrape course content."""
    print("Scraping course content...")
    
    # In a real implementation, we would need to navigate the course structure
    # For this example, we'll simulate a set of course pages
    course_pages = [
        {"url": f"{COURSE_CONTENT_URL}week1/lecture1.html", "title": "Introduction to TDS"},
        {"url": f"{COURSE_CONTENT_URL}week1/lecture2.html", "title": "Data Collection"},
        # Add more pages as needed
    ]
    
    results = []
    for page in course_pages:
        print(f"Fetching {page['url']}...")
        
        # In a real implementation, we would actually fetch the content
        # For this example, we'll create mock content since we can't access the actual website
        # html_content = await fetch_url_async(page["url"])
        
        # Mock content for illustration
        mock_html = f"""
        <html>
            <head><title>{page['title']}</title></head>
            <body>
                <h1 class="course-title">{page['title']}</h1>
                <div class="course-content">
                    <p>This is mock content for {page['title']}.</p>
                    <p>TDS Jan 2025 course material as of April 15, 2025.</p>
                    <p>Students are expected to understand data science concepts and apply them.</p>
                    <ul>
                        <li>Data collection and preprocessing</li>
                        <li>Exploratory data analysis</li>
                        <li>Model building and evaluation</li>
                    </ul>
                    <a href="/course_content/TDS_Jan_2025/materials/syllabus.pdf">Syllabus</a>
                </div>
            </body>
        </html>
        """
        
        # Parse content
        document = parse_course_content(mock_html, page["url"])
        results.append(document)
        
        # Be nice to the server
        await asyncio.sleep(1)
    
    print(f"Scraped {len(results)} course content pages")
    return results

async def scrape_discourse() -> List[Dict[str, Any]]:
    """Scrape Discourse forum posts."""
    print("Scraping Discourse forum posts...")
    
    # In a real implementation, we would need to paginate through topics
    # For this example, we'll simulate a set of topics
    
    # Date range for posts: 1 Jan 2025 - 14 Apr 2025
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 4, 14)
    
    # Mock topic IDs
    topic_ids = [12345, 12346, 12347, 12348, 12349]
    
    results = []
    for topic_id in topic_ids:
        topic_url = f"{DISCOURSE_BASE_URL}/t/{topic_id}.json"
        print(f"Fetching {topic_url}...")
        
        # In a real implementation, we would actually fetch the content
        # For this example, we'll create mock content
        # topic_json = json.loads(await fetch_url_async(topic_url, session_type="discourse"))
        
        # Mock topic data
        mock_topic = {
            "id": topic_id,
            "title": f"TDS Question about Assignment {topic_id % 5}",
            "slug": f"tds-question-about-assignment-{topic_id % 5}",
            "created_at": "2025-02-15T10:30:00Z",
            "post_stream": {
                "posts": [
                    {
                        "id": topic_id * 10 + 1,
                        "username": f"student{topic_id % 10}",
                        "created_at": "2025-02-15T10:30:00Z",
                        "cooked": f"<p>I'm having trouble with the TDS assignment {topic_id % 5}. Can someone help?</p><p>The error I'm getting is: ValueError: Model not found.</p>"
                    },
                    {
                        "id": topic_id * 10 + 2,
                        "username": "teaching_assistant",
                        "created_at": "2025-02-15T14:20:00Z",
                        "cooked": f"<p>For the TDS Jan 2025 course, you need to use the specific model mentioned in the assignment: gpt-3.5-turbo-0125.</p><p>Please check the API documentation and make sure you're using the correct model name.</p>"
                    }
                ]
            }
        }
        
        # Parse content
        document = parse_discourse_post(mock_topic)
        
        # Check if post date is within range
        post_date = document["metadata"]["date"]
        try:
            post_datetime = datetime.strptime(post_date, "%Y-%m-%d")
            if start_date <= post_datetime <= end_date:
                results.append(document)
        except ValueError:
            # If date parsing fails, include it anyway
            results.append(document)
        
        # Be nice to the server
        await asyncio.sleep(1)
    
    print(f"Scraped {len(results)} Discourse topics")
    return results

# If this script is run directly, test the scrapers
if __name__ == "__main__":
    init_scrapers()
    asyncio.run(scrape_course_content())
    asyncio.run(scrape_discourse())
