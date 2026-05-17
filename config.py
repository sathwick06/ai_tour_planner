"""
Configuration module for the Travel Planning Assistant
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables. Please set it in .env file.")

API_NINJAS_KEY = os.getenv("API_NINJAS_KEY")
if not API_NINJAS_KEY:
    raise ValueError("API_NINJAS_KEY not found in environment variables. Please set it in .env file.")

# Model Configuration
MODEL = "claude-sonnet-4-20250514"

# Application Configuration
APP_NAME = "Travel Planning Assistant"
DEFAULT_CHAT_MODEL = MODEL

# Agent Configuration
AGENT_CONFIG = {
    "intent_agent": {
        "temperature": 0.5,
        "max_tokens": 1000,
    },
    "research_agent": {
        "temperature": 0.6,
        "max_tokens": 1500,
    },
    "itinerary_agent": {
        "temperature": 0.7,
        "max_tokens": 2000,
    },
    "budget_agent": {
        "temperature": 0.5,
        "max_tokens": 1500,
    },
}

# Streamlit Configuration
STREAMLIT_CONFIG = {
    "page_title": APP_NAME,
    "page_icon": "✈️",
    "layout": "wide",
}
