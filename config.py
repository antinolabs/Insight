"""
Configuration for Prajna Insights
"""

import os
from datetime import timedelta

class Config:
    """Base configuration class."""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///instance/prajna_insights.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JWT_SECRET_KEY = SECRET_KEY
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)
    
    # File upload settings
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # LLM Configuration
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
    
    # Redis Configuration
    REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
    REDIS_DB = int(os.environ.get('REDIS_DB', 0))
    REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD')
    
    # Monitoring
    VERSION = '1.0.0'
    
    @staticmethod
    def init_app(app):
        """Initialize application with this config."""
        pass

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('DEV_DATABASE_URL') or 'sqlite:///instance/prajna_dev.db'

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///instance/prajna_prod.db'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}