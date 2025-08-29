"""
Shared authentication routes for OntExtract and OntServe
"""
from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app
from flask_login import login_user, logout_user, login_required, current_user
from datetime import datetime, timezone


def create_auth_blueprint(db, User, template_prefix='auth'):
    """Factory function to create authentication blueprint"""
    
    auth_bp = Blueprint('auth', __name__)

    @auth_bp.route('/login', methods=['GET', 'POST'])
    def login():
        """User login"""
        if current_user.is_authenticated:
            return redirect(url_for('index'))
        
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            remember = bool(request.form.get('remember'))
            
            if not username or not password:
                flash('Please enter both username and password', 'error')
                return render_template(f'{template_prefix}/login.html')
            
            user = User.query.filter_by(username=username).first()
            
            if user and user.check_password(password) and user.is_active:
                login_user(user, remember=remember)
                
                # Update last login with timezone-aware datetime
                user.last_login = datetime.now(timezone.utc)
                db.session.commit()
                
                # Log successful login
                current_app.logger.info(f"User {username} logged in successfully")
                
                next_page = request.args.get('next')
                if next_page:
                    return redirect(next_page)
                return redirect(url_for('index'))
            else:
                current_app.logger.warning(f"Failed login attempt for username: {username}")
                flash('Invalid username or password', 'error')
        
        return render_template(f'{template_prefix}/login.html')

    @auth_bp.route('/register', methods=['GET', 'POST'])
    def register():
        """User registration"""
        # Check if registration is enabled
        if not current_app.config.get('ALLOW_REGISTRATION', True):
            flash('Registration is currently disabled', 'error')
            return redirect(url_for('auth.login'))
            
        if current_user.is_authenticated:
            return redirect(url_for('index'))
        
        if request.method == 'POST':
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')
            confirm_password = request.form.get('confirm_password')
            first_name = request.form.get('first_name', '').strip()
            last_name = request.form.get('last_name', '').strip()
            organization = request.form.get('organization', '').strip()
            
            # Basic validation
            if not all([username, email, password, confirm_password]):
                flash('Username, email and password are required', 'error')
                return render_template(f'{template_prefix}/register.html')
            
            if password != confirm_password:
                flash('Passwords do not match', 'error')
                return render_template(f'{template_prefix}/register.html')
            
            if len(password) < 6:
                flash('Password must be at least 6 characters long', 'error')
                return render_template(f'{template_prefix}/register.html')
            
            # Check if username or email already exists
            if User.query.filter_by(username=username).first():
                flash('Username already exists', 'error')
                return render_template(f'{template_prefix}/register.html')
            
            if User.query.filter_by(email=email).first():
                flash('Email already registered', 'error')
                return render_template(f'{template_prefix}/register.html')
            
            # Create new user
            try:
                user = User(
                    username=username, 
                    email=email, 
                    password=password,
                    first_name=first_name or None,
                    last_name=last_name or None,
                    organization=organization or None
                )
                db.session.add(user)
                db.session.commit()
                
                current_app.logger.info(f"New user registered: {username}")
                flash('Registration successful! Please log in.', 'success')
                return redirect(url_for('auth.login'))
                
            except Exception as e:
                db.session.rollback()
                current_app.logger.error(f"Registration error for {username}: {e}")
                flash('An error occurred during registration', 'error')
                return render_template(f'{template_prefix}/register.html')
        
        return render_template(f'{template_prefix}/register.html')

    @auth_bp.route('/logout')
    @login_required
    def logout():
        """User logout"""
        username = current_user.username if current_user.is_authenticated else 'Unknown'
        logout_user()
        current_app.logger.info(f"User {username} logged out")
        flash('You have been logged out', 'info')
        return redirect(url_for('auth.login'))

    @auth_bp.route('/profile')
    @login_required
    def profile():
        """User profile page"""
        return render_template(f'{template_prefix}/profile.html')

    return auth_bp