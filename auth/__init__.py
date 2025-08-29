"""
Shared authentication module for OntExtract and OntServe
"""
from flask_login import LoginManager
from .models import create_user_model
from .routes import create_auth_blueprint


def setup_auth(app, db, template_prefix='auth', allow_registration=True):
    """
    Set up authentication for Flask application
    
    Args:
        app: Flask application instance
        db: SQLAlchemy database instance
        template_prefix: Prefix for template directory (default: 'auth')
        allow_registration: Whether to allow user registration (default: True)
        
    Returns:
        tuple: (login_manager, User_model, auth_blueprint)
    """
    
    # Configure application
    app.config['ALLOW_REGISTRATION'] = allow_registration
    
    # Initialize login manager
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'
    
    # Create User model
    User = create_user_model(db)
    
    @login_manager.user_loader
    def load_user(user_id):
        """Load user by ID for Flask-Login"""
        return db.session.get(User, int(user_id))
    
    # Create authentication blueprint
    auth_bp = create_auth_blueprint(db, User, template_prefix)
    
    return login_manager, User, auth_bp


def setup_cli_commands(app, db, User):
    """
    Set up CLI commands for user management
    
    Args:
        app: Flask application instance
        db: SQLAlchemy database instance
        User: User model class
    """
    import click
    from sqlalchemy import or_
    
    @app.cli.command("create-admin")
    @click.option("--username", prompt=True)
    @click.option("--email", prompt=True)
    @click.option("--password", prompt=True, hide_input=True, confirmation_prompt=True)
    @click.option("--first-name", default="")
    @click.option("--last-name", default="")
    @click.option("--organization", default="")
    def create_admin_command(username, email, password, first_name, last_name, organization):
        """Create an admin user."""
        with app.app_context():
            # Check if user already exists
            existing = db.session.query(User).filter(
                or_(User.username == username, User.email == email)
            ).first()
            if existing:
                click.echo("User with that username or email already exists.")
                return
                
            user = User(
                username=username, 
                email=email, 
                password=password, 
                is_admin=True,
                first_name=first_name or None,
                last_name=last_name or None,
                organization=organization or None
            )
            db.session.add(user)
            db.session.commit()
            click.echo(f"Admin user '{username}' created successfully.")

    @app.cli.command("list-users")
    def list_users_command():
        """List all users."""
        with app.app_context():
            users = User.query.all()
            if not users:
                click.echo("No users found.")
                return
                
            click.echo("Users:")
            for user in users:
                status = "✓" if user.is_active else "✗"
                admin = "(Admin)" if user.is_admin else ""
                click.echo(f"  {status} {user.username} <{user.email}> {admin}")

    @app.cli.command("deactivate-user")
    @click.option("--username", prompt=True)
    def deactivate_user_command(username):
        """Deactivate a user account."""
        with app.app_context():
            user = User.query.filter_by(username=username).first()
            if not user:
                click.echo(f"User '{username}' not found.")
                return
                
            user.is_active = False
            db.session.commit()
            click.echo(f"User '{username}' has been deactivated.")

    @app.cli.command("activate-user")
    @click.option("--username", prompt=True)
    def activate_user_command(username):
        """Activate a user account."""
        with app.app_context():
            user = User.query.filter_by(username=username).first()
            if not user:
                click.echo(f"User '{username}' not found.")
                return
                
            user.is_active = True
            db.session.commit()
            click.echo(f"User '{username}' has been activated.")


__all__ = ['setup_auth', 'setup_cli_commands']