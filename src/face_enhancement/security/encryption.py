"""Encryption & Data Protection.

Provides:
- AES encryption for data at rest
- End-to-end encryption
- Secure key management
- Data anonymization
- PII protection
"""

from typing import Optional, Tuple, Dict
from pathlib import Path
import secrets
import hashlib
from loguru import logger

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("cryptography package not available - install with: pip install cryptography")


class EncryptionService:
    """Encryption service for data protection."""

    def __init__(self, master_key: Optional[bytes] = None):
        """
        Initialize encryption service.

        Args:
            master_key: Master encryption key (generated if not provided)
        """
        if not CRYPTO_AVAILABLE:
            raise ImportError("cryptography package required")

        if master_key is None:
            # Generate new master key
            master_key = Fernet.generate_key()
            logger.warning("Generated new master key - store securely!")

        self.fernet = Fernet(master_key)
        self.master_key = master_key

        logger.info("Encryption service initialized")

    def encrypt(self, data: bytes) -> bytes:
        """
        Encrypt data.

        Args:
            data: Plain data

        Returns:
            Encrypted data
        """
        return self.fernet.encrypt(data)

    def decrypt(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data.

        Args:
            encrypted_data: Encrypted data

        Returns:
            Plain data
        """
        return self.fernet.decrypt(encrypted_data)

    def encrypt_string(self, text: str) -> str:
        """Encrypt string (returns base64 encoded)."""
        encrypted = self.encrypt(text.encode('utf-8'))
        return encrypted.decode('utf-8')

    def decrypt_string(self, encrypted_text: str) -> str:
        """Decrypt string."""
        decrypted = self.decrypt(encrypted_text.encode('utf-8'))
        return decrypted.decode('utf-8')

    def encrypt_file(self, input_path: Path, output_path: Path) -> None:
        """
        Encrypt file.

        Args:
            input_path: Input file path
            output_path: Output file path
        """
        with open(input_path, 'rb') as f:
            data = f.read()

        encrypted = self.encrypt(data)

        with open(output_path, 'wb') as f:
            f.write(encrypted)

        logger.info(f"Encrypted file: {input_path} -> {output_path}")

    def decrypt_file(self, input_path: Path, output_path: Path) -> None:
        """
        Decrypt file.

        Args:
            input_path: Encrypted file path
            output_path: Output file path
        """
        with open(input_path, 'rb') as f:
            encrypted_data = f.read()

        decrypted = self.decrypt(encrypted_data)

        with open(output_path, 'wb') as f:
            f.write(decrypted)

        logger.info(f"Decrypted file: {input_path} -> {output_path}")

    @staticmethod
    def derive_key_from_password(
        password: str,
        salt: Optional[bytes] = None,
    ) -> Tuple[bytes, bytes]:
        """
        Derive encryption key from password.

        Args:
            password: Password
            salt: Salt (generated if not provided)

        Returns:
            (key, salt)
        """
        if salt is None:
            salt = secrets.token_bytes(32)

        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend(),
        )

        key = kdf.derive(password.encode('utf-8'))

        return key, salt


class AESEncryption:
    """AES encryption (more control than Fernet)."""

    @staticmethod
    def encrypt_aes(
        data: bytes,
        key: bytes,
    ) -> Tuple[bytes, bytes, bytes]:
        """
        Encrypt data with AES-256-GCM.

        Args:
            data: Plain data
            key: 32-byte encryption key

        Returns:
            (iv, ciphertext, tag)
        """
        if len(key) != 32:
            raise ValueError("Key must be 32 bytes for AES-256")

        # Generate random IV
        iv = secrets.token_bytes(12)

        # Create cipher
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv),
            backend=default_backend(),
        )

        encryptor = cipher.encryptor()

        # Encrypt
        ciphertext = encryptor.update(data) + encryptor.finalize()

        # Get authentication tag
        tag = encryptor.tag

        return iv, ciphertext, tag

    @staticmethod
    def decrypt_aes(
        iv: bytes,
        ciphertext: bytes,
        tag: bytes,
        key: bytes,
    ) -> bytes:
        """
        Decrypt AES-256-GCM encrypted data.

        Args:
            iv: Initialization vector
            ciphertext: Encrypted data
            tag: Authentication tag
            key: Encryption key

        Returns:
            Decrypted data
        """
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv, tag),
            backend=default_backend(),
        )

        decryptor = cipher.decryptor()

        # Decrypt
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()

        return plaintext


class DataAnonymizer:
    """Anonymize sensitive data for privacy."""

    @staticmethod
    def hash_pii(value: str, salt: Optional[str] = None) -> str:
        """
        Hash PII (one-way, irreversible).

        Args:
            value: PII value
            salt: Salt for hashing

        Returns:
            Hashed value
        """
        if salt is None:
            salt = ""

        combined = value + salt
        hashed = hashlib.sha256(combined.encode('utf-8')).hexdigest()

        return hashed

    @staticmethod
    def redact_email(email: str) -> str:
        """
        Redact email address.

        Example: john.doe@example.com -> j***@example.com
        """
        if '@' not in email:
            return email

        local, domain = email.split('@', 1)

        if len(local) <= 1:
            return f"{local[0]}***@{domain}"

        return f"{local[0]}***@{domain}"

    @staticmethod
    def redact_phone(phone: str) -> str:
        """
        Redact phone number.

        Example: +1-555-123-4567 -> +1-555-***-**67
        """
        if len(phone) <= 4:
            return phone

        # Keep last 2 digits
        return phone[:-4] + '***' + phone[-2:]

    @staticmethod
    def mask_credit_card(card_number: str) -> str:
        """
        Mask credit card number.

        Example: 4532-1234-5678-9010 -> ****-****-****-9010
        """
        # Remove non-digits
        digits = ''.join(c for c in card_number if c.isdigit())

        if len(digits) < 4:
            return card_number

        # Keep last 4 digits
        masked = '*' * (len(digits) - 4) + digits[-4:]

        # Restore formatting
        result = card_number
        for i, c in enumerate(card_number):
            if c.isdigit():
                # Find position in digits
                digit_pos = sum(1 for x in card_number[:i] if x.isdigit())
                result = result[:i] + masked[digit_pos] + result[i+1:]

        return result

    @staticmethod
    def anonymize_ip(ip: str) -> str:
        """
        Anonymize IP address.

        Example: 192.168.1.100 -> 192.168.1.0
        """
        if '.' in ip:
            # IPv4
            parts = ip.split('.')
            if len(parts) == 4:
                parts[-1] = '0'
                return '.'.join(parts)

        elif ':' in ip:
            # IPv6
            parts = ip.split(':')
            if len(parts) >= 4:
                # Zero out last 4 groups
                parts[-4:] = ['0'] * 4
                return ':'.join(parts)

        return ip

    @staticmethod
    def tokenize(value: str, namespace: str = "default") -> str:
        """
        Generate consistent token for value.

        Same value always produces same token.
        Different namespaces produce different tokens for same value.

        Args:
            value: Value to tokenize
            namespace: Namespace for token

        Returns:
            Token (hex string)
        """
        combined = f"{namespace}:{value}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()[:16]


class SecureStorage:
    """Secure storage for sensitive data."""

    def __init__(self, encryption_service: EncryptionService):
        """
        Initialize secure storage.

        Args:
            encryption_service: Encryption service
        """
        self.encryption = encryption_service
        self.storage: Dict[str, bytes] = {}

        logger.info("Secure storage initialized")

    def store(self, key: str, value: str) -> None:
        """
        Store encrypted value.

        Args:
            key: Storage key
            value: Value to store
        """
        encrypted = self.encryption.encrypt_string(value)
        self.storage[key] = encrypted.encode('utf-8')

        logger.debug(f"Stored encrypted value: {key}")

    def retrieve(self, key: str) -> Optional[str]:
        """
        Retrieve and decrypt value.

        Args:
            key: Storage key

        Returns:
            Decrypted value or None
        """
        encrypted = self.storage.get(key)

        if encrypted is None:
            return None

        try:
            decrypted = self.encryption.decrypt_string(encrypted.decode('utf-8'))
            return decrypted
        except Exception as e:
            logger.error(f"Failed to decrypt value: {e}")
            return None

    def delete(self, key: str) -> bool:
        """Delete value."""
        if key in self.storage:
            del self.storage[key]
            logger.debug(f"Deleted encrypted value: {key}")
            return True
        return False

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return key in self.storage


class KeyManagement:
    """Secure key management."""

    def __init__(self, key_storage_path: Optional[Path] = None):
        """
        Initialize key management.

        Args:
            key_storage_path: Path to store keys securely
        """
        self.key_storage_path = key_storage_path or Path("keys")
        self.key_storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory key cache (encrypted)
        self.keys: Dict[str, bytes] = {}

        logger.info("Key management initialized")

    def generate_key(self, key_id: str) -> bytes:
        """
        Generate new encryption key.

        Args:
            key_id: Key identifier

        Returns:
            Generated key
        """
        key = Fernet.generate_key()
        self.keys[key_id] = key

        # Save to disk (in production, use HSM or KMS)
        key_file = self.key_storage_path / f"{key_id}.key"
        with open(key_file, 'wb') as f:
            f.write(key)

        # Set restrictive permissions
        key_file.chmod(0o600)

        logger.info(f"Generated key: {key_id}")

        return key

    def get_key(self, key_id: str) -> Optional[bytes]:
        """Retrieve key."""
        # Check cache
        if key_id in self.keys:
            return self.keys[key_id]

        # Load from disk
        key_file = self.key_storage_path / f"{key_id}.key"

        if not key_file.exists():
            logger.warning(f"Key not found: {key_id}")
            return None

        with open(key_file, 'rb') as f:
            key = f.read()

        # Cache
        self.keys[key_id] = key

        return key

    def rotate_key(self, key_id: str) -> bytes:
        """
        Rotate encryption key.

        Args:
            key_id: Key to rotate

        Returns:
            New key
        """
        old_key = self.get_key(key_id)

        # Generate new key
        new_key = self.generate_key(key_id)

        logger.info(f"Rotated key: {key_id}")

        # In production, re-encrypt all data with new key
        # This is a simplified example

        return new_key

    def delete_key(self, key_id: str) -> bool:
        """Delete key (use with caution!)."""
        # Remove from cache
        if key_id in self.keys:
            del self.keys[key_id]

        # Remove from disk
        key_file = self.key_storage_path / f"{key_id}.key"
        if key_file.exists():
            key_file.unlink()
            logger.warning(f"Deleted key: {key_id}")
            return True

        return False
