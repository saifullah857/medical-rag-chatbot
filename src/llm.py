from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
import os
from pathlib import Path

# Load .env (for local development)
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

# Try Streamlit Secrets first, then .env
try:
    api_key = st.secrets.get("GROQ_API_KEY")
except Exception:
    api_key = None

if not api_key:
    api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found in Streamlit Secrets or .env"
    )

llm = ChatGroq(
    api_key=api_key,
    model="qwen/qwen3-32b",
    temperature=0.1,
)