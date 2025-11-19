"""Input Validation & Sanitization.

Protects against:
- SQL injection
- XSS attacks
- Path traversal
- Command injection
- Buffer overflows
- Malicious file uploads
"""

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import re
import magic
import hashlib
from loguru import logger


class ValidationError(Exception):
    """Validation error."""
    pass


class InputValidator:
    """Comprehensive input validation."""

    # Allowed image MIME types
    ALLOWED_IMAGE_TYPES = {
        'image/jpeg',
        'image/png',
        'image/webp',
        'image/bmp',
        'image/tiff',
    }

    # Allowed video MIME types
    ALLOWED_VIDEO_TYPES = {
        'video/mp4',
        'video/mpeg',
        'video/quicktime',
        'video/x-msvideo',
        'video/webm',
    }

    # Maximum file sizes (bytes)
    MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50 MB
    MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500 MB

    # Dangerous file extensions
    DANGEROUS_EXTENSIONS = {
        'exe', 'bat', 'cmd', 'sh', 'ps1', 'vbs', 'js',
        'jar', 'py', 'rb', 'pl', 'php', 'asp', 'aspx',
        'dll', 'so', 'dylib', 'scr', 'pif', 'app',
    }

    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validate email address.

        Args:
            email: Email to validate

        Returns:
            True if valid

        Raises:
            ValidationError if invalid
        """
        # RFC 5322 simplified pattern
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

        if not re.match(pattern, email):
            raise ValidationError(f"Invalid email address: {email}")

        if len(email) > 254:
            raise ValidationError("Email too long")

        return True

    @staticmethod
    def validate_password(password: str, min_length: int = 8) -> bool:
        """
        Validate password strength.

        Requirements:
        - Minimum length
        - At least one uppercase letter
        - At least one lowercase letter
        - At least one digit
        - At least one special character

        Args:
            password: Password to validate
            min_length: Minimum password length

        Returns:
            True if valid

        Raises:
            ValidationError if invalid
        """
        if len(password) < min_length:
            raise ValidationError(f"Password must be at least {min_length} characters")

        if not re.search(r'[A-Z]', password):
            raise ValidationError("Password must contain uppercase letter")

        if not re.search(r'[a-z]', password):
            raise ValidationError("Password must contain lowercase letter")

        if not re.search(r'[0-9]', password):
            raise ValidationError("Password must contain digit")

        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            raise ValidationError("Password must contain special character")

        return True

    @staticmethod
    def validate_file_upload(
        file_path: Path,
        allowed_types: Optional[set] = None,
        max_size: Optional[int] = None,
    ) -> Tuple[str, int]:
        """
        Validate uploaded file.

        Checks:
        - File exists
        - MIME type (magic bytes, not extension)
        - File size
        - No dangerous extensions
        - Not executable

        Args:
            file_path: Path to file
            allowed_types: Allowed MIME types
            max_size: Maximum file size in bytes

        Returns:
            (mime_type, file_size)

        Raises:
            ValidationError if validation fails
        """
        # Check file exists
        if not file_path.exists():
            raise ValidationError(f"File not found: {file_path}")

        # Check file size
        file_size = file_path.stat().st_size

        if max_size and file_size > max_size:
            raise ValidationError(
                f"File too large: {file_size} bytes (max: {max_size})"
            )

        if file_size == 0:
            raise ValidationError("Empty file")

        # Check extension
        extension = file_path.suffix.lower().lstrip('.')
        if extension in InputValidator.DANGEROUS_EXTENSIONS:
            raise ValidationError(f"Dangerous file extension: {extension}")

        # Check MIME type using magic bytes
        try:
            mime = magic.Magic(mime=True)
            mime_type = mime.from_file(str(file_path))
        except Exception as e:
            logger.warning(f"Could not determine MIME type: {e}")
            mime_type = "application/octet-stream"

        if allowed_types and mime_type not in allowed_types:
            raise ValidationError(
                f"Invalid file type: {mime_type}. "
                f"Allowed: {', '.join(allowed_types)}"
            )

        # Check if executable (Unix)
        if file_path.stat().st_mode & 0o111:
            raise ValidationError("Executable files not allowed")

        return mime_type, file_size

    @staticmethod
    def validate_image_file(file_path: Path) -> Tuple[str, int]:
        """Validate image file upload."""
        return InputValidator.validate_file_upload(
            file_path,
            allowed_types=InputValidator.ALLOWED_IMAGE_TYPES,
            max_size=InputValidator.MAX_IMAGE_SIZE,
        )

    @staticmethod
    def validate_video_file(file_path: Path) -> Tuple[str, int]:
        """Validate video file upload."""
        return InputValidator.validate_file_upload(
            file_path,
            allowed_types=InputValidator.ALLOWED_VIDEO_TYPES,
            max_size=InputValidator.MAX_VIDEO_SIZE,
        )

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename to prevent path traversal.

        Args:
            filename: Original filename

        Returns:
            Sanitized filename
        """
        # Remove path separators
        filename = filename.replace('/', '').replace('\\', '')

        # Remove null bytes
        filename = filename.replace('\x00', '')

        # Remove leading dots
        filename = filename.lstrip('.')

        # Keep only alphanumeric, dash, underscore, dot
        filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)

        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = name[:250] + '.' + ext if ext else name[:255]

        if not filename:
            raise ValidationError("Invalid filename")

        return filename

    @staticmethod
    def validate_uuid(uuid_str: str) -> bool:
        """Validate UUID format."""
        pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'

        if not re.match(pattern, uuid_str.lower()):
            raise ValidationError(f"Invalid UUID: {uuid_str}")

        return True

    @staticmethod
    def validate_url(url: str, allowed_schemes: Optional[List[str]] = None) -> bool:
        """
        Validate URL.

        Args:
            url: URL to validate
            allowed_schemes: Allowed URL schemes (default: http, https)

        Returns:
            True if valid

        Raises:
            ValidationError if invalid
        """
        if allowed_schemes is None:
            allowed_schemes = ['http', 'https']

        # Basic URL pattern
        pattern = r'^(https?://)[a-zA-Z0-9.-]+(/[^?#]*)?(\?[^#]*)?(#.*)?$'

        if not re.match(pattern, url):
            raise ValidationError(f"Invalid URL: {url}")

        # Check scheme
        scheme = url.split('://')[0]
        if scheme not in allowed_schemes:
            raise ValidationError(
                f"Invalid URL scheme: {scheme}. "
                f"Allowed: {', '.join(allowed_schemes)}"
            )

        # Check for SSRF attacks (localhost, private IPs)
        domain = url.split('://')[1].split('/')[0].split(':')[0]

        blocked_domains = {
            'localhost',
            '127.0.0.1',
            '0.0.0.0',
            '169.254.169.254',  # AWS metadata
            'metadata.google.internal',  # GCP metadata
        }

        if domain.lower() in blocked_domains:
            raise ValidationError(f"Blocked domain: {domain}")

        # Check for private IP ranges
        if domain.replace('.', '').isdigit():
            # Simple check for common private ranges
            if (domain.startswith('10.') or
                domain.startswith('192.168.') or
                domain.startswith('172.')):
                raise ValidationError(f"Private IP not allowed: {domain}")

        return True

    @staticmethod
    def validate_json_payload(
        payload: Dict[str, Any],
        required_fields: List[str],
        max_depth: int = 10,
        max_size: int = 1024 * 1024,  # 1 MB
    ) -> bool:
        """
        Validate JSON payload.

        Args:
            payload: JSON payload
            required_fields: Required field names
            max_depth: Maximum nesting depth
            max_size: Maximum payload size

        Returns:
            True if valid

        Raises:
            ValidationError if invalid
        """
        import json

        # Check size
        payload_str = json.dumps(payload)
        if len(payload_str) > max_size:
            raise ValidationError("Payload too large")

        # Check required fields
        for field in required_fields:
            if field not in payload:
                raise ValidationError(f"Missing required field: {field}")

        # Check depth
        def check_depth(obj, current_depth=0):
            if current_depth > max_depth:
                raise ValidationError("Payload too deeply nested")

            if isinstance(obj, dict):
                for value in obj.values():
                    check_depth(value, current_depth + 1)
            elif isinstance(obj, list):
                for item in obj:
                    check_depth(item, current_depth + 1)

        check_depth(payload)

        return True

    @staticmethod
    def sanitize_sql(value: str) -> str:
        """
        Sanitize value for SQL (though parameterized queries preferred).

        Args:
            value: Value to sanitize

        Returns:
            Sanitized value
        """
        # Escape single quotes
        value = value.replace("'", "''")

        # Remove SQL keywords (basic check)
        dangerous_keywords = [
            'DROP', 'DELETE', 'UPDATE', 'INSERT', 'EXEC',
            'EXECUTE', 'SCRIPT', 'UNION', 'SELECT', '--',
        ]

        for keyword in dangerous_keywords:
            # Case-insensitive removal
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            if pattern.search(value):
                raise ValidationError(f"Potentially dangerous SQL keyword: {keyword}")

        return value

    @staticmethod
    def validate_age(age: int) -> bool:
        """Validate age value."""
        if not isinstance(age, int):
            raise ValidationError("Age must be integer")

        if age < 0 or age > 150:
            raise ValidationError(f"Invalid age: {age}")

        return True

    @staticmethod
    def validate_confidence(confidence: float) -> bool:
        """Validate confidence score."""
        if not isinstance(confidence, (int, float)):
            raise ValidationError("Confidence must be numeric")

        if confidence < 0.0 or confidence > 1.0:
            raise ValidationError(f"Confidence must be 0-1: {confidence}")

        return True


class RateLimiter:
    """Rate limiting to prevent abuse."""

    def __init__(self):
        """Initialize rate limiter."""
        # Store request counts: {key: [(timestamp, count), ...]}
        self.requests: Dict[str, List[Tuple[float, int]]] = {}

        logger.info("Rate limiter initialized")

    def check_rate_limit(
        self,
        key: str,
        max_requests: int,
        window_seconds: int = 60,
    ) -> bool:
        """
        Check if request is within rate limit.

        Args:
            key: Rate limit key (user_id, IP, etc.)
            max_requests: Maximum requests allowed
            window_seconds: Time window in seconds

        Returns:
            True if within limit

        Raises:
            ValidationError if rate limit exceeded
        """
        import time

        now = time.time()
        window_start = now - window_seconds

        # Get request history for key
        if key not in self.requests:
            self.requests[key] = []

        # Remove old requests outside window
        self.requests[key] = [
            (ts, count) for ts, count in self.requests[key]
            if ts > window_start
        ]

        # Count requests in window
        total_requests = sum(count for _, count in self.requests[key])

        if total_requests >= max_requests:
            raise ValidationError(
                f"Rate limit exceeded: {total_requests}/{max_requests} "
                f"requests in {window_seconds}s"
            )

        # Add current request
        self.requests[key].append((now, 1))

        return True

    def get_remaining(
        self,
        key: str,
        max_requests: int,
        window_seconds: int = 60,
    ) -> int:
        """Get remaining requests in current window."""
        import time

        now = time.time()
        window_start = now - window_seconds

        if key not in self.requests:
            return max_requests

        # Count requests in window
        recent_requests = [
            count for ts, count in self.requests[key]
            if ts > window_start
        ]

        total = sum(recent_requests)
        return max(0, max_requests - total)


class SecurityScanner:
    """Security scanning utilities."""

    @staticmethod
    def scan_for_malware_signatures(data: bytes) -> bool:
        """
        Basic malware signature detection.

        Args:
            data: File data

        Returns:
            True if clean

        Raises:
            ValidationError if malware detected
        """
        # Known malware signatures (simplified)
        dangerous_patterns = [
            b'MZ\x90\x00',  # PE executable header
            b'#!/bin/bash',  # Shell script
            b'#!/bin/sh',
            b'<script>',  # JavaScript
            b'eval(',
            b'exec(',
            b'system(',
            b'`',  # Command execution
        ]

        for pattern in dangerous_patterns:
            if pattern in data:
                raise ValidationError("Potentially malicious content detected")

        return True

    @staticmethod
    def compute_file_hash(file_path: Path, algorithm: str = 'sha256') -> str:
        """
        Compute file hash for integrity verification.

        Args:
            file_path: Path to file
            algorithm: Hash algorithm

        Returns:
            Hex digest of hash
        """
        hasher = hashlib.new(algorithm)

        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)

        return hasher.hexdigest()
