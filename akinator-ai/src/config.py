"""
Configuration module for AI Akinator.

Loads environment variables and provides centralized configuration.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    anthropic_api_key: str = Field(..., env='ANTHROPIC_API_KEY')
    langchain_api_key: str = Field(..., env='LANGCHAIN_API_KEY')
    
    # LangSmith Configuration
    langchain_tracing_v2: bool = Field(True, env='LANGCHAIN_TRACING_V2')
    langchain_project: str = Field('akinator-ai', env='LANGCHAIN_PROJECT')
    langchain_endpoint: str = Field(
        'https://api.smith.langchain.com',
        env='LANGCHAIN_ENDPOINT'
    )
    
    # Game Configuration
    max_questions: int = Field(20, env='MAX_QUESTIONS')
    confidence_threshold: float = Field(0.85, env='CONFIDENCE_THRESHOLD')
    
    # Model Configuration
    model_name: str = Field(
        'claude-sonnet-4-20250514',
        env='MODEL_NAME'
    )
    model_temperature: float = Field(0.0, env='MODEL_TEMPERATURE')
    
    # Paths
    knowledge_base_path: str = Field(
        './knowledge_base',
        env='KNOWLEDGE_BASE_PATH'
    )
    
    # Logging
    log_level: str = Field('INFO', env='LOG_LEVEL')
    
    class Config:
        """Pydantic configuration."""
        env_file = '.env'
        env_file_encoding = 'utf-8'
        case_sensitive = False


# Global settings instance
try:
    settings = Settings()
except Exception as e:
    print(f"⚠️  Error loading settings: {e}")
    print("💡 Make sure you have created a .env file with required API keys")
    print("   Copy .env.example to .env and fill in your API keys")
    raise


# Derived configurations
PROJECT_ROOT = Path(__file__).parent.parent
KNOWLEDGE_BASE_PATH = PROJECT_ROOT / settings.knowledge_base_path

# Validate paths
if not KNOWLEDGE_BASE_PATH.exists():
    print(f"📁 Creating knowledge base directory: {KNOWLEDGE_BASE_PATH}")
    KNOWLEDGE_BASE_PATH.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Get application settings."""
    return settings


def validate_environment() -> bool:
    """
    Validate that all required environment variables are set.
    
    Returns:
        bool: True if environment is valid, False otherwise
    """
    try:
        # Check for required API keys
        if not settings.anthropic_api_key:
            print("❌ ANTHROPIC_API_KEY not set")
            return False
        
        if not settings.langchain_api_key:
            print("❌ LANGCHAIN_API_KEY not set")
            return False
        
        # Validate paths
        if not KNOWLEDGE_BASE_PATH.exists():
            print(f"❌ Knowledge base path does not exist: {KNOWLEDGE_BASE_PATH}")
            return False
        
        print("✅ Environment validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Environment validation failed: {e}")
        return False


if __name__ == "__main__":
    """Test configuration loading."""
    print("🔧 Testing configuration...")
    print(f"📊 Max questions: {settings.max_questions}")
    print(f"🎯 Confidence threshold: {settings.confidence_threshold}")
    print(f"🤖 Model: {settings.model_name}")
    print(f"📁 Knowledge base: {KNOWLEDGE_BASE_PATH}")
    print(f"🔬 LangSmith project: {settings.langchain_project}")
    
    if validate_environment():
        print("\n✅ Configuration loaded successfully!")
    else:
        print("\n❌ Configuration validation failed")
