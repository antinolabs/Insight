"""
Prajna Insights - Business Analytics Platform
Flask Application Factory
"""

from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager
import sys
import os
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config as config_dict

# Initialize extensions
db = SQLAlchemy()
migrate = Migrate()
jwt = JWTManager()

def create_app(config_name='default'):
    """
    Application factory function.
    
    Args:
        config_name (str): Configuration name (development, production, testing)
    
    Returns:
        Flask: Configured Flask application instance
    """
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(config_dict[config_name])
    config_dict[config_name].init_app(app)
    
    # Initialize extensions with app
    db.init_app(app)
    migrate.init_app(app, db)
    jwt.init_app(app)
    CORS(app, origins=app.config.get('CORS_ORIGINS', '*'))
    
    # Configure JWT
    app.config['JWT_SECRET_KEY'] = app.config.get('SECRET_KEY', 'dev-secret-key')
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = 3600
    
    # Register blueprints
    from app.api.auth import auth_bp
    from app.api.data import data_bp
    from app.ui.routes import ui_bp
    
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(data_bp, url_prefix='/api/data')
    app.register_blueprint(ui_bp)
    
    # Initialize cache middleware
    try:
        from app.middleware.cache_layer import cache_middleware
        cache_middleware.init_app(app)
        print("✅ Cache middleware initialized successfully")
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize cache middleware: {e}")
        # Continue execution - caching is optional
    
    # Initialize security middleware
    try:
        from app.security.middleware import security_middleware, rate_limit
        security_middleware.init_app(app)
        print("✅ Security middleware initialized successfully")
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize security middleware: {e}")
        # Continue execution - security is optional
        rate_limit = lambda x: lambda f: f  # Fallback decorator
    
    # Initialize monitoring
    try:
        from app.monitoring.setup import monitoring_setup
        monitoring_setup.init_app(app)
        print("✅ Production monitoring initialized successfully")
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize monitoring: {e}")
        # Continue execution - monitoring is optional
    
    # Create database tables
    with app.app_context():
        db.create_all()
        
        # Add production optimizations
        try:
            from app.enhancements.database_optimizer import enhance_database, add_performance_indexes
            enhance_database(app)
            add_performance_indexes()
            print("✅ Database optimizations applied successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not apply all database optimizations: {e}")
            # Continue execution - optimizations are optional
    
    # Health check endpoints
    @app.route('/health')
    def health_check():
        """Basic health check endpoint."""
        return {
            'status': 'healthy',
            'service': 'Prajna Insights',
            'version': '1.0.0'
        }
    
    @app.route('/health/live')
    def liveness():
        """Kubernetes liveness probe endpoint."""
        return {'status': 'alive'}, 200
    
    @app.route('/health/ready')
    def readiness():
        """Kubernetes readiness probe endpoint."""
        try:
            from app.monitoring.health import health_checker
            health_result = health_checker.run_health_check()
            
            is_ready = health_result['status'] in ['healthy', 'warning']
            status_code = 200 if is_ready else 503
            
            return jsonify(health_result), status_code
            
        except Exception as e:
            return jsonify({
                'status': 'unhealthy',
                'error': str(e),
                'checked_at': datetime.utcnow().isoformat()
            }), 503
    
    @app.route('/health/detailed')
    def detailed_health():
        """Detailed health check with all components."""
        try:
            from app.monitoring.health import health_checker
            return jsonify(health_checker.run_health_check())
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error': str(e),
                'checked_at': datetime.utcnow().isoformat()
            }), 500
    
    @app.route('/metrics')
    def metrics():
        """Prometheus metrics endpoint."""
        try:
            from app.monitoring.setup import monitoring_setup
            if monitoring_setup.metrics:
                return monitoring_setup.metrics.generate_latest()
            else:
                return "# Prometheus metrics not available", 503
        except Exception as e:
            return f"# Error generating metrics: {e}", 500
    
    # Cache management endpoints
    @app.route('/api/cache/stats')
    def cache_stats():
        """Get cache statistics."""
        try:
            from app.middleware.cache_layer import cache_middleware
            return cache_middleware.get_cache_stats()
        except Exception as e:
            return {'status': 'error', 'message': str(e)}, 500
    
    @app.route('/api/cache/clear', methods=['POST'])
    @rate_limit('cache_clear')
    def clear_cache():
        """Clear all cache entries."""
        try:
            from app.middleware.cache_layer import cache_middleware
            cache_middleware.clear_all_cache()
            return {'status': 'success', 'message': 'Cache cleared successfully'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}, 500
    
    # Root endpoint
    @app.route('/')
    def index():
        """Root endpoint with API information."""
        return {
            'message': 'Welcome to Prajna Insights',
            'description': 'Business Analytics Platform with AI-Powered Insights',
            'version': '1.0.0',
            'endpoints': {
                'health': '/health',
                'auth': '/api/auth',
                'data': '/api/data'
            }
        }
    
    return app
