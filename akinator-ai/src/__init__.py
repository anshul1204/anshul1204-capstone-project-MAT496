"""
AI Akinator - Multi-Agent Guessing Game

A sophisticated guessing game powered by LangGraph multi-agent system
with LangSmith learning capabilities.
"""

__version__ = '0.1.0'
__author__ = 'Your Name'

# Make key components available at package level
from src.config import settings, get_settings

__all__ = ['settings', 'get_settings', '__version__']
