"""Security Module Tests."""

import pytest
from pathlib import Path
import tempfile

from src.face_enhancement.security import (
    AuthenticationService,
    Role,
    Permission,
    InputValidator,
    ValidationError,
    RateLimiter,
    EncryptionService,
    DataAnonymizer,
)


class TestAuthentication:
    """Test authentication functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.auth_service = AuthenticationService(secret_key="test-secret-key-12345")

    def test_register_user(self):
        """Test user registration."""
        user = self.auth_service.register_user(
            email="test@example.com",
            password="SecurePass123!",
            tenant_id="tenant-001",
            role=Role.USER,
        )

        assert user.email == "test@example.com"
        assert user.tenant_id == "tenant-001"
        assert user.role == Role.USER
        assert user.active is True

    def test_duplicate_email_fails(self):
        """Test that duplicate email registration fails."""
        self.auth_service.register_user(
            email="test@example.com",
            password="SecurePass123!",
            tenant_id="tenant-001",
        )

        with pytest.raises(ValueError, match="Email already registered"):
            self.auth_service.register_user(
                email="test@example.com",
                password="SecurePass123!",
                tenant_id="tenant-001",
            )

    def test_login_success(self):
        """Test successful login."""
        self.auth_service.register_user(
            email="test@example.com",
            password="SecurePass123!",
            tenant_id="tenant-001",
        )

        result = self.auth_service.login("test@example.com", "SecurePass123!")

        assert result is not None
        assert "access_token" in result
        assert "refresh_token" in result
        assert result["token_type"] == "Bearer"

    def test_login_wrong_password(self):
        """Test login with wrong password."""
        self.auth_service.register_user(
            email="test@example.com",
            password="SecurePass123!",
            tenant_id="tenant-001",
        )

        result = self.auth_service.login("test@example.com", "WrongPassword")

        assert result is None

    def test_jwt_token_verification(self):
        """Test JWT token verification."""
        user = self.auth_service.register_user(
            email="test@example.com",
            password="SecurePass123!",
            tenant_id="tenant-001",
        )

        token = self.auth_service.jwt_auth.generate_token(user)
        verified_user = self.auth_service.authenticate_token(token)

        assert verified_user is not None
        assert verified_user.user_id == user.user_id


class TestInputValidation:
    """Test input validation."""

    def test_validate_email_success(self):
        """Test valid email."""
        assert InputValidator.validate_email("test@example.com") is True

    def test_validate_email_invalid(self):
        """Test invalid email."""
        with pytest.raises(ValidationError):
            InputValidator.validate_email("invalid-email")

    def test_validate_password_success(self):
        """Test valid password."""
        assert InputValidator.validate_password("SecurePass123!") is True

    def test_validate_password_too_short(self):
        """Test password too short."""
        with pytest.raises(ValidationError, match="at least"):
            InputValidator.validate_password("Short1!")

    def test_validate_password_no_uppercase(self):
        """Test password without uppercase."""
        with pytest.raises(ValidationError, match="uppercase"):
            InputValidator.validate_password("securepass123!")

    def test_sanitize_filename(self):
        """Test filename sanitization."""
        dirty = "../../../etc/passwd"
        clean = InputValidator.sanitize_filename(dirty)

        assert "/" not in clean
        assert ".." not in clean

    def test_validate_file_upload(self):
        """Test file upload validation."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
            f.write(b"\xff\xd8\xff")  # JPEG magic bytes
            temp_path = Path(f.name)

        try:
            mime_type, size = InputValidator.validate_file_upload(
                temp_path,
                allowed_types={"image/jpeg"},
                max_size=1024 * 1024,
            )

            assert size > 0

        finally:
            temp_path.unlink()


class TestRateLimiter:
    """Test rate limiting."""

    def test_rate_limit_enforcement(self):
        """Test rate limit is enforced."""
        limiter = RateLimiter()

        # Should allow first 5 requests
        for i in range(5):
            assert limiter.check_rate_limit(
                key="test-user",
                max_requests=5,
                window_seconds=60,
            ) is True

        # 6th request should fail
        with pytest.raises(ValidationError, match="Rate limit exceeded"):
            limiter.check_rate_limit(
                key="test-user",
                max_requests=5,
                window_seconds=60,
            )

    def test_rate_limit_different_keys(self):
        """Test different keys have separate limits."""
        limiter = RateLimiter()

        # Max out user1
        for i in range(5):
            limiter.check_rate_limit("user1", max_requests=5)

        # user2 should still work
        assert limiter.check_rate_limit("user2", max_requests=5) is True


class TestEncryption:
    """Test encryption."""

    def test_encrypt_decrypt_string(self):
        """Test string encryption and decryption."""
        encryptor = EncryptionService()

        original = "Sensitive data"
        encrypted = encryptor.encrypt_string(original)
        decrypted = encryptor.decrypt_string(encrypted)

        assert encrypted != original
        assert decrypted == original

    def test_encrypt_decrypt_bytes(self):
        """Test bytes encryption and decryption."""
        encryptor = EncryptionService()

        original = b"Binary data"
        encrypted = encryptor.encrypt(original)
        decrypted = encryptor.decrypt(encrypted)

        assert encrypted != original
        assert decrypted == original


class TestDataAnonymization:
    """Test data anonymization."""

    def test_redact_email(self):
        """Test email redaction."""
        email = "john.doe@example.com"
        redacted = DataAnonymizer.redact_email(email)

        assert redacted == "j***@example.com"

    def test_mask_credit_card(self):
        """Test credit card masking."""
        card = "4532-1234-5678-9010"
        masked = DataAnonymizer.mask_credit_card(card)

        assert "9010" in masked
        assert "4532" not in masked

    def test_hash_pii(self):
        """Test PII hashing."""
        value = "sensitive-id"
        hashed1 = DataAnonymizer.hash_pii(value)
        hashed2 = DataAnonymizer.hash_pii(value)

        # Same input -> same hash
        assert hashed1 == hashed2

        # Different input -> different hash
        hashed3 = DataAnonymizer.hash_pii("different-value")
        assert hashed1 != hashed3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
