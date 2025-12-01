"""
Setup script for AI Akinator package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="akinator-ai",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered Akinator game using LangGraph multi-agent system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/akinator-ai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "langgraph>=0.2.0",
        "langchain>=0.3.0",
        "langchain-anthropic>=0.2.0",
        "langsmith>=0.1.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "rich>=13.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "web": [
            "streamlit>=1.28.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "akinator=main:main",
        ],
    },
)
