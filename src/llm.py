from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load .env first
load_dotenv(".env")

# Get the API key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

# Set environment variable (optional but recommended)
os.environ["GROQ_API_KEY"] = api_key

llm = ChatGroq(
    api_key = api_key,
    model_name="qwen/qwen3-32b",
    temperature=0.7,
    max_tokens=1024
)