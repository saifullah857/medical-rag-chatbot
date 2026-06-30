from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
import os
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"

print("Env path:", env_path)
print("Exists:", env_path.exists())

load_dotenv(dotenv_path=env_path)

try:
    api_key = st.secrets["GROQ_API_KEY"]
    print("Using Streamlit Secret")
except Exception:
    api_key = os.getenv("GROQ_API_KEY")
    print("Using .env")

print("API Key:", api_key)

if not api_key:
    raise ValueError("API Key not found")

llm = ChatGroq(
    api_key=api_key,
    model="qwen/qwen3-32b",
    temperature=0.7,
)