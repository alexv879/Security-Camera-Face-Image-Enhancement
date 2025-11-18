"""Compliance and audit features for enterprise deployments.

Provides:
- GDPR compliance toolkit
- SOC 2 compliance features
- HIPAA compliance support
- Comprehensive audit logging
- Data retention policies
- Consent management
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib
from loguru import logger


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"
    SOC2 = "soc2"
    HIPAA = "hipaa"
    CCPA = "ccpa"
    ISO27001 = "iso27001"


class DataCategory(Enum):
    """Categories of data for privacy management."""
    PII = "personally_identifiable_information"
    PHI = "protected_health_information"
    BIOMETRIC = "biometric_data"
    FINANCIAL = "financial_data"
    LOCATION = "location_data"


@dataclass
class ConsentRecord:
    """Record of user consent."""

    user_id: str
    tenant_id: str
    purpose: str  # What the data is used for
    granted: bool
    granted_at: datetime
    expires_at: Optional[datetime] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    consent_text: str = ""

    def is_valid(self) -> bool:
        """Check if consent is still valid."""
        if not self.granted:
            return False
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        return True


@dataclass
class AuditLogEntry:
    """Comprehensive audit log entry."""

    timestamp: datetime
    tenant_id: str
    user_id: str
    action: str  # What was done
    resource_type: str  # What was affected
    resource_id: str
    ip_address: str
    user_agent: str
    status: str  # success, failure, unauthorized
    details: Dict[str, Any] = field(default_factory=dict)
    data_category: Optional[DataCategory] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "action": self.action,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "ip_address": self._anonymize_ip(self.ip_address),
            "user_agent": self.user_agent,
            "status": self.status,
            "details": self.details,
            "data_category": self.data_category.value if self.data_category else None,
        }

    @staticmethod
    def _anonymize_ip(ip: str) -> str:
        """Anonymize IP address for privacy."""
        # Keep first 3 octets for IPv4, mask last octet
        parts = ip.split('.')
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.{parts[2]}.XXX"
        return hashlib.sha256(ip.encode()).hexdigest()[:16]


class AuditLogger:
    """
    Comprehensive audit logging system.

    Required for SOC 2, HIPAA, and GDPR compliance.
    """

    def __init__(self, storage_path: str = "./audit_logs"):
        """Initialize audit logger."""
        self.storage_path = storage_path
        self.logs: List[AuditLogEntry] = []

        logger.info(f"Audit logger initialized: {storage_path}")

    def log_action(
        self,
        tenant_id: str,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        ip_address: str,
        user_agent: str,
        status: str = "success",
        details: Optional[Dict] = None,
        data_category: Optional[DataCategory] = None,
    ) -> AuditLogEntry:
        """
        Log an action for audit trail.

        Args:
            tenant_id: Tenant ID
            user_id: User ID
            action: Action performed
            resource_type: Type of resource
            resource_id: Resource ID
            ip_address: Client IP
            user_agent: Client user agent
            status: Action status
            details: Additional details
            data_category: Data category if applicable

        Returns:
            Audit log entry
        """
        entry = AuditLogEntry(
            timestamp=datetime.now(),
            tenant_id=tenant_id,
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            user_agent=user_agent,
            status=status,
            details=details or {},
            data_category=data_category,
        )

        self.logs.append(entry)
        self._persist_log(entry)

        return entry

    def get_user_activity(
        self,
        tenant_id: str,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[AuditLogEntry]:
        """Get all activity for a user (GDPR requirement)."""
        logs = [
            log for log in self.logs
            if log.tenant_id == tenant_id and log.user_id == user_id
        ]

        if start_date:
            logs = [log for log in logs if log.timestamp >= start_date]
        if end_date:
            logs = [log for log in logs if log.timestamp <= end_date]

        return logs

    def get_data_access_logs(
        self,
        tenant_id: str,
        data_category: DataCategory,
        days: int = 30,
    ) -> List[AuditLogEntry]:
        """Get all access to specific data category."""
        cutoff = datetime.now() - timedelta(days=days)

        return [
            log for log in self.logs
            if log.tenant_id == tenant_id
            and log.data_category == data_category
            and log.timestamp >= cutoff
        ]

    def export_audit_trail(
        self,
        tenant_id: str,
        start_date: datetime,
        end_date: datetime,
        output_path: str,
    ) -> None:
        """Export audit trail (compliance requirement)."""
        logs = [
            log for log in self.logs
            if log.tenant_id == tenant_id
            and start_date <= log.timestamp <= end_date
        ]

        # Export as JSON
        with open(output_path, 'w') as f:
            json.dump([log.to_dict() for log in logs], f, indent=2)

        logger.info(f"Exported {len(logs)} audit entries to {output_path}")

    def _persist_log(self, entry: AuditLogEntry) -> None:
        """Persist log entry to durable storage."""
        # Simplified - real implementation would write to database or log aggregator
        pass


class GDPRCompliance:
    """
    GDPR compliance toolkit.

    Provides tools for:
    - Right to access (Article 15)
    - Right to erasure (Article 17)
    - Right to data portability (Article 20)
    - Consent management
    - Data breach notification
    """

    def __init__(self, audit_logger: AuditLogger):
        """Initialize GDPR compliance."""
        self.audit_logger = audit_logger
        self.consents: Dict[str, List[ConsentRecord]] = {}

        logger.info("GDPR compliance initialized")

    def record_consent(
        self,
        user_id: str,
        tenant_id: str,
        purpose: str,
        granted: bool,
        consent_text: str,
        expires_in_days: Optional[int] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> ConsentRecord:
        """
        Record user consent (GDPR Article 7).

        Args:
            user_id: User ID
            tenant_id: Tenant ID
            purpose: Purpose of data processing
            granted: Whether consent was granted
            consent_text: Full text of consent
            expires_in_days: Expiration in days
            ip_address: User IP
            user_agent: User agent

        Returns:
            Consent record
        """
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)

        consent = ConsentRecord(
            user_id=user_id,
            tenant_id=tenant_id,
            purpose=purpose,
            granted=granted,
            granted_at=datetime.now(),
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent,
            consent_text=consent_text,
        )

        key = f"{tenant_id}:{user_id}"
        if key not in self.consents:
            self.consents[key] = []
        self.consents[key].append(consent)

        # Audit log
        self.audit_logger.log_action(
            tenant_id=tenant_id,
            user_id=user_id,
            action="consent_recorded",
            resource_type="consent",
            resource_id=purpose,
            ip_address=ip_address or "unknown",
            user_agent=user_agent or "unknown",
            status="success",
            details={"purpose": purpose, "granted": granted},
            data_category=DataCategory.PII,
        )

        return consent

    def check_consent(
        self, user_id: str, tenant_id: str, purpose: str
    ) -> bool:
        """Check if user has valid consent for purpose."""
        key = f"{tenant_id}:{user_id}"
        consents = self.consents.get(key, [])

        # Get most recent consent for purpose
        relevant_consents = [c for c in consents if c.purpose == purpose]
        if not relevant_consents:
            return False

        latest_consent = max(relevant_consents, key=lambda c: c.granted_at)
        return latest_consent.is_valid()

    def export_user_data(
        self, user_id: str, tenant_id: str, output_path: str
    ) -> Dict:
        """
        Export all user data (Right to Access, Article 15).

        Args:
            user_id: User ID
            tenant_id: Tenant ID
            output_path: Output file path

        Returns:
            Summary of exported data
        """
        # Collect all data about user
        user_data = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "export_date": datetime.now().isoformat(),
            "consents": [],
            "audit_logs": [],
            "processing_records": [],
        }

        # Consents
        key = f"{tenant_id}:{user_id}"
        consents = self.consents.get(key, [])
        user_data["consents"] = [
            {
                "purpose": c.purpose,
                "granted": c.granted,
                "granted_at": c.granted_at.isoformat(),
                "expires_at": c.expires_at.isoformat() if c.expires_at else None,
            }
            for c in consents
        ]

        # Audit logs
        logs = self.audit_logger.get_user_activity(tenant_id, user_id)
        user_data["audit_logs"] = [log.to_dict() for log in logs]

        # Export
        with open(output_path, 'w') as f:
            json.dump(user_data, f, indent=2)

        logger.info(f"Exported user data for {user_id} to {output_path}")

        return {
            "total_records": len(user_data["audit_logs"]),
            "consents": len(user_data["consents"]),
            "export_path": output_path,
        }

    def delete_user_data(
        self, user_id: str, tenant_id: str
    ) -> Dict:
        """
        Delete all user data (Right to Erasure, Article 17).

        Args:
            user_id: User ID
            tenant_id: Tenant ID

        Returns:
            Summary of deletion
        """
        deleted = {
            "consents": 0,
            "audit_logs": 0,
            "processing_records": 0,
        }

        # Delete consents
        key = f"{tenant_id}:{user_id}"
        if key in self.consents:
            deleted["consents"] = len(self.consents[key])
            del self.consents[key]

        # Note: Audit logs may need to be retained for legal/compliance
        # Mark them as deleted but don't physically delete

        # Log the deletion
        self.audit_logger.log_action(
            tenant_id=tenant_id,
            user_id="system",
            action="user_data_deleted",
            resource_type="user",
            resource_id=user_id,
            ip_address="internal",
            user_agent="system",
            status="success",
            details=deleted,
            data_category=DataCategory.PII,
        )

        logger.info(f"Deleted user data for {user_id}")

        return deleted


class HIPAACompliance:
    """
    HIPAA compliance support for healthcare applications.

    Provides:
    - PHI access controls
    - Encryption requirements
    - Audit trails
    - BAA (Business Associate Agreement) support
    """

    def __init__(self, audit_logger: AuditLogger):
        """Initialize HIPAA compliance."""
        self.audit_logger = audit_logger
        logger.info("HIPAA compliance initialized")

    def log_phi_access(
        self,
        tenant_id: str,
        user_id: str,
        patient_id: str,
        action: str,
        ip_address: str,
        user_agent: str,
    ) -> None:
        """Log access to Protected Health Information."""
        self.audit_logger.log_action(
            tenant_id=tenant_id,
            user_id=user_id,
            action=action,
            resource_type="phi",
            resource_id=patient_id,
            ip_address=ip_address,
            user_agent=user_agent,
            status="success",
            data_category=DataCategory.PHI,
        )

    def generate_access_report(
        self,
        tenant_id: str,
        patient_id: str,
        days: int = 30,
    ) -> List[AuditLogEntry]:
        """
        Generate access report for patient (required by HIPAA).

        Args:
            tenant_id: Tenant ID
            patient_id: Patient ID
            days: Number of days to include

        Returns:
            List of access log entries
        """
        cutoff = datetime.now() - timedelta(days=days)

        logs = [
            log for log in self.audit_logger.logs
            if log.tenant_id == tenant_id
            and log.resource_id == patient_id
            and log.data_category == DataCategory.PHI
            and log.timestamp >= cutoff
        ]

        return logs

    def check_minimum_necessary(
        self, requested_fields: List[str], purpose: str
    ) -> List[str]:
        """
        Ensure minimum necessary standard (HIPAA).

        Only return fields necessary for purpose.

        Args:
            requested_fields: Fields requested
            purpose: Purpose of access

        Returns:
            Filtered list of allowed fields
        """
        # Simplified - real implementation would have detailed policies
        allowed_by_purpose = {
            "treatment": ["*"],  # All fields for treatment
            "payment": ["patient_id", "billing_info"],
            "operations": ["patient_id", "date", "provider"],
        }

        allowed = allowed_by_purpose.get(purpose, [])

        if "*" in allowed:
            return requested_fields

        return [f for f in requested_fields if f in allowed]


class SOC2Compliance:
    """
    SOC 2 compliance features.

    Provides controls for:
    - Security
    - Availability
    - Processing Integrity
    - Confidentiality
    - Privacy
    """

    def __init__(self, audit_logger: AuditLogger):
        """Initialize SOC 2 compliance."""
        self.audit_logger = audit_logger
        self.change_logs: List[Dict] = []

        logger.info("SOC 2 compliance initialized")

    def log_configuration_change(
        self,
        tenant_id: str,
        user_id: str,
        component: str,
        old_value: Any,
        new_value: Any,
        ip_address: str,
    ) -> None:
        """Log configuration changes (required for SOC 2)."""
        change = {
            "timestamp": datetime.now().isoformat(),
            "tenant_id": tenant_id,
            "user_id": user_id,
            "component": component,
            "old_value": str(old_value),
            "new_value": str(new_value),
            "ip_address": ip_address,
        }

        self.change_logs.append(change)

        self.audit_logger.log_action(
            tenant_id=tenant_id,
            user_id=user_id,
            action="configuration_changed",
            resource_type="configuration",
            resource_id=component,
            ip_address=ip_address,
            user_agent="system",
            status="success",
            details=change,
        )

    def generate_availability_report(
        self, start_date: datetime, end_date: datetime
    ) -> Dict:
        """Generate availability report for SOC 2."""
        # Simplified - real implementation would query monitoring system
        return {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "uptime_percentage": 99.95,
            "total_downtime_minutes": 21.6,
            "incidents": [],
        }


class DataRetentionPolicy:
    """
    Data retention policy management.

    Automatically delete data after retention period expires.
    """

    def __init__(self):
        """Initialize data retention policy."""
        self.policies: Dict[str, int] = {
            # Resource type -> days to retain
            "audit_logs": 2555,  # 7 years for compliance
            "user_data": 1825,  # 5 years
            "processing_records": 365,  # 1 year
            "temp_files": 7,  # 1 week
        }

        logger.info("Data retention policy initialized")

    def set_retention_period(
        self, resource_type: str, days: int
    ) -> None:
        """Set retention period for resource type."""
        self.policies[resource_type] = days
        logger.info(f"Set retention period for {resource_type}: {days} days")

    def get_retention_period(self, resource_type: str) -> int:
        """Get retention period for resource type."""
        return self.policies.get(resource_type, 365)  # Default 1 year

    def should_delete(
        self, resource_type: str, created_at: datetime
    ) -> bool:
        """Check if resource should be deleted based on retention policy."""
        retention_days = self.get_retention_period(resource_type)
        cutoff = datetime.now() - timedelta(days=retention_days)

        return created_at < cutoff
