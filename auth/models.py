"""
Shared authentication models for OntExtract and OntServe
"""
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timezone

class UserMixin(UserMixin):
    """Base user model that can be extended by both applications"""
    
    # Core user fields
    id = None  # Should be db.Column(db.Integer, primary_key=True) in subclass
    username = None  # Should be db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = None  # Should be db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = None  # Should be db.Column(db.String(256), nullable=False)
    
    # Profile information
    first_name = None  # Should be db.Column(db.String(50))
    last_name = None  # Should be db.Column(db.String(50))
    organization = None  # Should be db.Column(db.String(100))
    
    # Account status
    is_active = None  # Should be db.Column(db.Boolean, default=True, nullable=False)
    is_admin = None  # Should be db.Column(db.Boolean, default=False, nullable=False)
    
    # Timestamps
    created_at = None  # Should be db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = None  # Should be db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = None  # Should be db.Column(db.DateTime)
    
    def __init__(self, username, email, password, **kwargs):
        self.username = username
        self.email = email
        self.set_password(password)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def set_password(self, password):
        """Hash and set the user's password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if provided password matches the hash"""
        return check_password_hash(self.password_hash, password)
    
    def get_full_name(self):
        """Get the user's full name"""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.username
    
    def to_dict(self):
        """Convert user to dictionary for API responses"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.get_full_name(),
            'organization': self.organization,
            'is_active': self.is_active,
            'is_admin': self.is_admin,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }
    
    def __repr__(self):
        return f'<User {self.username}>'


def create_user_model(db):
    """Factory function to create a User model with the given database instance"""
    
    class User(UserMixin, db.Model):
        """User model for authentication and session management"""
        
        __tablename__ = 'users'
        
        id = db.Column(db.Integer, primary_key=True)
        username = db.Column(db.String(80), unique=True, nullable=False, index=True)
        email = db.Column(db.String(120), unique=True, nullable=False, index=True)
        password_hash = db.Column(db.String(256), nullable=False)
        
        # Profile information
        first_name = db.Column(db.String(50))
        last_name = db.Column(db.String(50))
        organization = db.Column(db.String(100))
        
        # Account status
        is_active = db.Column(db.Boolean, default=True, nullable=False)
        is_admin = db.Column(db.Boolean, default=False, nullable=False)
        
        # Timestamps - Use timezone-aware datetime
        created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
        updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
        last_login = db.Column(db.DateTime)
    
    return User