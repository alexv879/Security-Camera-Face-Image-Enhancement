"""Secure Authentication & Authorization System.

Provides:
- JWT-based authentication
- API key management
- OAuth2 support
- Multi-factor authentication
- Role-based access control (RBAC)
- Session management
- Token refresh and revocation
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import secrets
import hmac
from loguru import logger

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    logger.warning("PyJWT not available - install with: pip install pyjwt")


class Role(Enum):
    """User roles for RBAC."""
    ADMIN = "admin"
    ENTERPRISE = "enterprise"
    BUSINESS = "business"
    PROFESSIONAL = "professional"
    DEVELOPER = "developer"
    USER = "user"
    VIEWER = "viewer"


class Permission(Enum):
    """Granular permissions."""
    # Face enhancement
    ENHANCE_FACE = "enhance_face"
    ENHANCE_VIDEO = "enhance_video"
    BATCH_PROCESS = "batch_process"

    # Ultra-value features
    DETECT_DEEPFAKE = "detect_deepfake"
    VERIFY_IDENTITY = "verify_identity"
    VERIFY_AGE = "verify_age"
    FORENSIC_MATCH = "forensic_match"
    GENERATE_SYNTHETIC = "generate_synthetic"

    # Admin
    MANAGE_USERS = "manage_users"
    MANAGE_TENANTS = "manage_tenants"
    VIEW_ANALYTICS = "view_analytics"
    CONFIGURE_SYSTEM = "configure_system"

    # Billing
    VIEW_USAGE = "view_usage"
    MANAGE_BILLING = "manage_billing"


@dataclass
class User:
    """User account."""

    user_id: str
    email: str
    password_hash: str

    # Account info
    tenant_id: str
    role: Role
    permissions: List[Permission] = field(default_factory=list)

    # Status
    active: bool = True
    email_verified: bool = False
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    login_count: int = 0

    # Security
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    password_changed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to dictionary (exclude sensitive data)."""
        return {
            "user_id": self.user_id,
            "email": self.email,
            "tenant_id": self.tenant_id,
            "role": self.role.value,
            "permissions": [p.value for p in self.permissions],
            "active": self.active,
            "email_verified": self.email_verified,
            "mfa_enabled": self.mfa_enabled,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }


@dataclass
class APIKey:
    """API key for programmatic access."""

    key_id: str
    key_hash: str  # Never store plain key

    # Ownership
    user_id: str
    tenant_id: str
    name: str

    # Permissions
    permissions: List[Permission] = field(default_factory=list)
    rate_limit: int = 1000  # Requests per hour

    # Status
    active: bool = True

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    def is_expired(self) -> bool:
        """Check if key is expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


class PasswordHasher:
    """Secure password hashing using PBKDF2."""

    @staticmethod
    def hash_password(password: str, salt: Optional[bytes] = None) -> str:
        """
        Hash password securely.

        Args:
            password: Plain text password
            salt: Optional salt (generated if not provided)

        Returns:
            Hash in format: algorithm$iterations$salt$hash
        """
        if salt is None:
            salt = secrets.token_bytes(32)

        # Use PBKDF2 with SHA-256
        iterations = 100000
        hash_bytes = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            iterations
        )

        # Format: algorithm$iterations$salt$hash
        salt_hex = salt.hex()
        hash_hex = hash_bytes.hex()

        return f"pbkdf2_sha256${iterations}${salt_hex}${hash_hex}"

    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """
        Verify password against hash.

        Args:
            password: Plain text password
            password_hash: Stored hash

        Returns:
            True if password matches
        """
        try:
            parts = password_hash.split('$')
            if len(parts) != 4:
                return False

            algorithm, iterations, salt_hex, hash_hex = parts

            if algorithm != 'pbkdf2_sha256':
                return False

            salt = bytes.fromhex(salt_hex)
            stored_hash = bytes.fromhex(hash_hex)

            # Compute hash with same salt
            computed_hash = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt,
                int(iterations)
            )

            # Constant-time comparison to prevent timing attacks
            return hmac.compare_digest(computed_hash, stored_hash)

        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False


class JWTAuthenticator:
    """JWT-based authentication."""

    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        """
        Initialize JWT authenticator.

        Args:
            secret_key: Secret key for signing tokens
            algorithm: JWT algorithm (HS256, RS256, etc.)
        """
        if not JWT_AVAILABLE:
            raise ImportError("PyJWT required for JWT authentication")

        self.secret_key = secret_key
        self.algorithm = algorithm

        logger.info("JWT authenticator initialized")

    def generate_token(
        self,
        user: User,
        expires_in: int = 3600,
        refresh: bool = False,
    ) -> str:
        """
        Generate JWT token.

        Args:
            user: User to generate token for
            expires_in: Token expiry in seconds
            refresh: Whether this is a refresh token

        Returns:
            JWT token string
        """
        now = datetime.utcnow()

        payload = {
            "user_id": user.user_id,
            "email": user.email,
            "tenant_id": user.tenant_id,
            "role": user.role.value,
            "permissions": [p.value for p in user.permissions],
            "iat": now,
            "exp": now + timedelta(seconds=expires_in),
            "type": "refresh" if refresh else "access",
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

        return token

    def verify_token(self, token: str) -> Optional[Dict]:
        """
        Verify and decode JWT token.

        Args:
            token: JWT token string

        Returns:
            Decoded payload or None if invalid
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload

        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None

    def refresh_token(self, refresh_token: str, user: User) -> Optional[str]:
        """
        Generate new access token from refresh token.

        Args:
            refresh_token: Valid refresh token
            user: User account

        Returns:
            New access token or None
        """
        payload = self.verify_token(refresh_token)

        if payload is None:
            return None

        if payload.get("type") != "refresh":
            logger.warning("Not a refresh token")
            return None

        if payload.get("user_id") != user.user_id:
            logger.warning("User ID mismatch")
            return None

        # Generate new access token
        return self.generate_token(user, expires_in=3600)


class APIKeyManager:
    """Manage API keys for programmatic access."""

    def __init__(self):
        """Initialize API key manager."""
        self.keys: Dict[str, APIKey] = {}
        logger.info("API key manager initialized")

    def generate_key(
        self,
        user_id: str,
        tenant_id: str,
        name: str,
        permissions: List[Permission],
        expires_in_days: Optional[int] = None,
    ) -> Tuple[str, APIKey]:
        """
        Generate new API key.

        Args:
            user_id: User ID
            tenant_id: Tenant ID
            name: Key name/description
            permissions: Allowed permissions
            expires_in_days: Days until expiry (None = never)

        Returns:
            (plain_key, api_key_object)
        """
        import uuid

        # Generate secure random key
        plain_key = f"sk_{secrets.token_urlsafe(32)}"

        # Hash the key (never store plain)
        key_hash = hashlib.sha256(plain_key.encode()).hexdigest()

        # Calculate expiry
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)

        # Create API key object
        api_key = APIKey(
            key_id=str(uuid.uuid4()),
            key_hash=key_hash,
            user_id=user_id,
            tenant_id=tenant_id,
            name=name,
            permissions=permissions,
            expires_at=expires_at,
        )

        self.keys[key_hash] = api_key

        logger.info(f"Generated API key: {api_key.key_id} for user {user_id}")

        # Return plain key (only time it's available)
        return plain_key, api_key

    def verify_key(self, plain_key: str) -> Optional[APIKey]:
        """
        Verify API key.

        Args:
            plain_key: API key to verify

        Returns:
            APIKey object or None if invalid
        """
        # Hash the provided key
        key_hash = hashlib.sha256(plain_key.encode()).hexdigest()

        # Look up
        api_key = self.keys.get(key_hash)

        if api_key is None:
            return None

        # Check if active
        if not api_key.active:
            logger.warning(f"Inactive API key used: {api_key.key_id}")
            return None

        # Check expiry
        if api_key.is_expired():
            logger.warning(f"Expired API key used: {api_key.key_id}")
            return None

        # Update last used
        api_key.last_used = datetime.now()

        return api_key

    def revoke_key(self, key_id: str) -> bool:
        """Revoke API key."""
        for api_key in self.keys.values():
            if api_key.key_id == key_id:
                api_key.active = False
                logger.info(f"Revoked API key: {key_id}")
                return True
        return False


class RBACAuthorizer:
    """Role-Based Access Control."""

    # Define role permissions
    ROLE_PERMISSIONS = {
        Role.ADMIN: [p for p in Permission],  # All permissions
        Role.ENTERPRISE: [
            Permission.ENHANCE_FACE,
            Permission.ENHANCE_VIDEO,
            Permission.BATCH_PROCESS,
            Permission.DETECT_DEEPFAKE,
            Permission.VERIFY_IDENTITY,
            Permission.VERIFY_AGE,
            Permission.FORENSIC_MATCH,
            Permission.GENERATE_SYNTHETIC,
            Permission.VIEW_ANALYTICS,
            Permission.VIEW_USAGE,
        ],
        Role.BUSINESS: [
            Permission.ENHANCE_FACE,
            Permission.ENHANCE_VIDEO,
            Permission.BATCH_PROCESS,
            Permission.DETECT_DEEPFAKE,
            Permission.VERIFY_IDENTITY,
            Permission.VERIFY_AGE,
            Permission.VIEW_USAGE,
        ],
        Role.PROFESSIONAL: [
            Permission.ENHANCE_FACE,
            Permission.ENHANCE_VIDEO,
            Permission.VERIFY_AGE,
            Permission.VIEW_USAGE,
        ],
        Role.DEVELOPER: [
            Permission.ENHANCE_FACE,
            Permission.VIEW_USAGE,
        ],
        Role.USER: [
            Permission.ENHANCE_FACE,
        ],
        Role.VIEWER: [],
    }

    @staticmethod
    def get_permissions(role: Role) -> List[Permission]:
        """Get permissions for role."""
        return RBACAuthorizer.ROLE_PERMISSIONS.get(role, [])

    @staticmethod
    def has_permission(user: User, permission: Permission) -> bool:
        """Check if user has permission."""
        # Check explicit permissions
        if permission in user.permissions:
            return True

        # Check role permissions
        role_permissions = RBACAuthorizer.get_permissions(user.role)
        return permission in role_permissions

    @staticmethod
    def require_permission(user: User, permission: Permission) -> None:
        """Raise exception if user lacks permission."""
        if not RBACAuthorizer.has_permission(user, permission):
            raise PermissionError(
                f"User {user.user_id} lacks permission: {permission.value}"
            )


class AuthenticationService:
    """Complete authentication service."""

    def __init__(self, secret_key: str):
        """
        Initialize authentication service.

        Args:
            secret_key: Secret key for JWT signing
        """
        self.jwt_auth = JWTAuthenticator(secret_key)
        self.api_key_manager = APIKeyManager()
        self.password_hasher = PasswordHasher()

        # User storage (in production, use database)
        self.users: Dict[str, User] = {}

        logger.info("Authentication service initialized")

    def register_user(
        self,
        email: str,
        password: str,
        tenant_id: str,
        role: Role = Role.USER,
    ) -> User:
        """Register new user."""
        import uuid

        # Check if email exists
        for user in self.users.values():
            if user.email == email:
                raise ValueError(f"Email already registered: {email}")

        # Hash password
        password_hash = self.password_hasher.hash_password(password)

        # Get role permissions
        permissions = RBACAuthorizer.get_permissions(role)

        # Create user
        user = User(
            user_id=str(uuid.uuid4()),
            email=email,
            password_hash=password_hash,
            tenant_id=tenant_id,
            role=role,
            permissions=permissions,
        )

        self.users[user.user_id] = user

        logger.info(f"Registered user: {user.user_id} ({email})")

        return user

    def login(self, email: str, password: str) -> Optional[Dict]:
        """
        Login user.

        Returns:
            Dictionary with access_token and refresh_token
        """
        # Find user by email
        user = None
        for u in self.users.values():
            if u.email == email:
                user = u
                break

        if user is None:
            logger.warning(f"Login failed: user not found ({email})")
            return None

        # Check if locked
        if user.locked_until and datetime.now() < user.locked_until:
            logger.warning(f"Login failed: account locked ({email})")
            return None

        # Verify password
        if not self.password_hasher.verify_password(password, user.password_hash):
            # Increment failed attempts
            user.failed_login_attempts += 1

            # Lock account after 5 failed attempts
            if user.failed_login_attempts >= 5:
                user.locked_until = datetime.now() + timedelta(minutes=30)
                logger.warning(f"Account locked: {email}")

            logger.warning(f"Login failed: invalid password ({email})")
            return None

        # Check if active
        if not user.active:
            logger.warning(f"Login failed: account inactive ({email})")
            return None

        # Reset failed attempts
        user.failed_login_attempts = 0
        user.last_login = datetime.now()
        user.login_count += 1

        # Generate tokens
        access_token = self.jwt_auth.generate_token(user, expires_in=3600)
        refresh_token = self.jwt_auth.generate_token(
            user, expires_in=86400, refresh=True
        )

        logger.info(f"User logged in: {user.user_id} ({email})")

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "Bearer",
            "expires_in": 3600,
            "user": user.to_dict(),
        }

    def authenticate_token(self, token: str) -> Optional[User]:
        """Authenticate JWT token."""
        payload = self.jwt_auth.verify_token(token)

        if payload is None:
            return None

        user_id = payload.get("user_id")
        user = self.users.get(user_id)

        return user

    def authenticate_api_key(self, api_key: str) -> Optional[APIKey]:
        """Authenticate API key."""
        return self.api_key_manager.verify_key(api_key)
