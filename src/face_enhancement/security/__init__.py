"""Security Module.

Comprehensive security features:
- Authentication (JWT, API keys, OAuth2)
- Authorization (RBAC)
- Input validation and sanitization
- Rate limiting
- Encryption (at rest and in transit)
- Data anonymization
- Secure key management

Protection against:
- SQL injection
- XSS attacks
- CSRF
- Path traversal
- Command injection
- DDoS
- Unauthorized access
- Data breaches
"""

from .authentication import (
    AuthenticationService,
    User,
    APIKey,
    Role,
    Permission,
    JWTAuthenticator,
    APIKeyManager,
    RBACAuthorizer,
    PasswordHasher,
)

from .validation import (
    InputValidator,
    ValidationError,
    RateLimiter,
    SecurityScanner,
)

from .encryption import (
    EncryptionService,
    AESEncryption,
    DataAnonymizer,
    SecureStorage,
    KeyManagement,
)

__all__ = [
    # Authentication
    "AuthenticationService",
    "User",
    "APIKey",
    "Role",
    "Permission",
    "JWTAuthenticator",
    "APIKeyManager",
    "RBACAuthorizer",
    "PasswordHasher",
    # Validation
    "InputValidator",
    "ValidationError",
    "RateLimiter",
    "SecurityScanner",
    # Encryption
    "EncryptionService",
    "AESEncryption",
    "DataAnonymizer",
    "SecureStorage",
    "KeyManagement",
]
