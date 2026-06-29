from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
import os

# Load .env for local development
load_dotenv()

# Try Streamlit Secrets first
try:              
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Add it to Streamlit Secrets or your local .env file."
    )

llm = ChatGroq(
    api_key=api_key,
    model="qwen/qwen3-32b",
    temperature=0.7,
    max_tokens=1024,
)