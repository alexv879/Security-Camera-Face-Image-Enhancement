"""Multi-tenant SaaS platform implementation.

Provides:
- Tenant isolation and management
- Resource quotas and limits
- Usage tracking per tenant
- Tenant-specific configurations
- White-labeling support
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import uuid
from loguru import logger


class TenantTier(Enum):
    """Tenant subscription tiers."""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    BUSINESS = "business"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


@dataclass
class TenantQuota:
    """Resource quotas for a tenant."""

    # API limits
    requests_per_minute: int = 10
    requests_per_day: int = 1000
    requests_per_month: int = 10000

    # Processing limits
    max_image_size_mb: int = 10
    max_video_duration_seconds: int = 60
    max_concurrent_requests: int = 2

    # Storage limits
    storage_gb: int = 1

    # Feature access
    models_allowed: List[str] = field(default_factory=lambda: ["gfpgan"])
    advanced_features: bool = False
    custom_models: bool = False
    api_access: bool = True
    web_ui_access: bool = True

    # Support
    support_tier: str = "community"  # community, email, priority, dedicated
    sla_uptime: float = 99.5  # percentage


# Quota configurations by tier
TIER_QUOTAS = {
    TenantTier.FREE: TenantQuota(
        requests_per_minute=5,
        requests_per_day=100,
        requests_per_month=1000,
        max_image_size_mb=5,
        max_concurrent_requests=1,
        storage_gb=0,
        models_allowed=["gfpgan"],
        advanced_features=False,
        support_tier="community",
        sla_uptime=99.0,
    ),
    TenantTier.STARTER: TenantQuota(
        requests_per_minute=10,
        requests_per_day=1000,
        requests_per_month=10000,
        max_image_size_mb=10,
        max_concurrent_requests=2,
        storage_gb=1,
        models_allowed=["gfpgan", "codeformer"],
        advanced_features=False,
        support_tier="email",
        sla_uptime=99.5,
    ),
    TenantTier.PROFESSIONAL: TenantQuota(
        requests_per_minute=50,
        requests_per_day=10000,
        requests_per_month=100000,
        max_image_size_mb=20,
        max_video_duration_seconds=300,
        max_concurrent_requests=5,
        storage_gb=10,
        models_allowed=["gfpgan", "codeformer", "real_esrgan"],
        advanced_features=True,
        support_tier="priority",
        sla_uptime=99.9,
    ),
    TenantTier.BUSINESS: TenantQuota(
        requests_per_minute=200,
        requests_per_day=100000,
        requests_per_month=1000000,
        max_image_size_mb=50,
        max_video_duration_seconds=1800,
        max_concurrent_requests=20,
        storage_gb=100,
        models_allowed=["*"],  # All models
        advanced_features=True,
        custom_models=True,
        support_tier="priority",
        sla_uptime=99.95,
    ),
    TenantTier.ENTERPRISE: TenantQuota(
        requests_per_minute=1000,
        requests_per_day=-1,  # Unlimited
        requests_per_month=-1,
        max_image_size_mb=200,
        max_video_duration_seconds=-1,
        max_concurrent_requests=100,
        storage_gb=1000,
        models_allowed=["*"],
        advanced_features=True,
        custom_models=True,
        support_tier="dedicated",
        sla_uptime=99.99,
    ),
}


@dataclass
class Tenant:
    """Represents a tenant in the multi-tenant system."""

    tenant_id: str
    name: str
    tier: TenantTier
    created_at: datetime = field(default_factory=datetime.now)

    # Contact info
    admin_email: str = ""
    company: str = ""

    # Status
    is_active: bool = True
    is_trial: bool = False
    trial_ends_at: Optional[datetime] = None

    # Configuration
    quota: TenantQuota = field(default_factory=TenantQuota)
    custom_domain: Optional[str] = None
    white_label_config: Dict = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = asdict(self)
        data["tier"] = self.tier.value
        data["created_at"] = self.created_at.isoformat()
        if self.trial_ends_at:
            data["trial_ends_at"] = self.trial_ends_at.isoformat()
        return data


class TenantManager:
    """
    Manages multi-tenant operations.

    Provides:
    - Tenant CRUD operations
    - Quota enforcement
    - Usage tracking
    - Data isolation
    """

    def __init__(self, storage_backend: str = "memory"):
        """
        Initialize tenant manager.

        Args:
            storage_backend: Storage backend (memory, redis, postgres)
        """
        self.storage_backend = storage_backend
        self.tenants: Dict[str, Tenant] = {}
        self.usage_cache: Dict[str, Dict] = {}

        logger.info(f"Tenant manager initialized with {storage_backend} backend")

    def create_tenant(
        self,
        name: str,
        tier: TenantTier,
        admin_email: str,
        company: str = "",
        is_trial: bool = False,
        trial_days: int = 14,
    ) -> Tenant:
        """
        Create a new tenant.

        Args:
            name: Tenant name
            tier: Subscription tier
            admin_email: Admin email
            company: Company name
            is_trial: Whether this is a trial
            trial_days: Trial duration in days

        Returns:
            Created tenant
        """
        tenant_id = str(uuid.uuid4())

        # Get quota for tier
        quota = TIER_QUOTAS[tier]

        # Set trial end date if trial
        trial_ends_at = None
        if is_trial:
            trial_ends_at = datetime.now() + timedelta(days=trial_days)

        tenant = Tenant(
            tenant_id=tenant_id,
            name=name,
            tier=tier,
            admin_email=admin_email,
            company=company,
            quota=quota,
            is_trial=is_trial,
            trial_ends_at=trial_ends_at,
        )

        self.tenants[tenant_id] = tenant
        self._persist_tenant(tenant)

        logger.info(f"Created tenant: {name} (ID: {tenant_id}, Tier: {tier.value})")

        return tenant

    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID."""
        return self.tenants.get(tenant_id)

    def update_tenant_tier(self, tenant_id: str, new_tier: TenantTier) -> bool:
        """
        Update tenant subscription tier.

        Args:
            tenant_id: Tenant ID
            new_tier: New tier

        Returns:
            True if successful
        """
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return False

        old_tier = tenant.tier
        tenant.tier = new_tier
        tenant.quota = TIER_QUOTAS[new_tier]

        # End trial if upgrading
        if tenant.is_trial and new_tier != TenantTier.FREE:
            tenant.is_trial = False
            tenant.trial_ends_at = None

        self._persist_tenant(tenant)

        logger.info(f"Updated tenant {tenant_id} tier: {old_tier.value} -> {new_tier.value}")

        return True

    def check_quota(
        self,
        tenant_id: str,
        quota_type: str,
        amount: int = 1,
    ) -> bool:
        """
        Check if tenant has quota available.

        Args:
            tenant_id: Tenant ID
            quota_type: Type of quota (requests_per_minute, etc.)
            amount: Amount to check

        Returns:
            True if quota available
        """
        tenant = self.get_tenant(tenant_id)
        if not tenant or not tenant.is_active:
            return False

        # Check if trial expired
        if tenant.is_trial and tenant.trial_ends_at:
            if datetime.now() > tenant.trial_ends_at:
                logger.warning(f"Tenant {tenant_id} trial expired")
                return False

        # Get current usage
        usage = self._get_current_usage(tenant_id, quota_type)

        # Get limit
        limit = getattr(tenant.quota, quota_type, -1)

        # -1 means unlimited
        if limit == -1:
            return True

        # Check if over quota
        return (usage + amount) <= limit

    def record_usage(
        self,
        tenant_id: str,
        usage_type: str,
        amount: int = 1,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Record usage for a tenant.

        Args:
            tenant_id: Tenant ID
            usage_type: Type of usage
            amount: Amount to record
            metadata: Optional metadata
        """
        if tenant_id not in self.usage_cache:
            self.usage_cache[tenant_id] = {
                "requests_minute": 0,
                "requests_day": 0,
                "requests_month": 0,
                "last_minute": datetime.now().minute,
                "last_day": datetime.now().day,
                "last_month": datetime.now().month,
            }

        usage = self.usage_cache[tenant_id]

        # Reset counters if time period changed
        now = datetime.now()
        if now.minute != usage["last_minute"]:
            usage["requests_minute"] = 0
            usage["last_minute"] = now.minute
        if now.day != usage["last_day"]:
            usage["requests_day"] = 0
            usage["last_day"] = now.day
        if now.month != usage["last_month"]:
            usage["requests_month"] = 0
            usage["last_month"] = now.month

        # Increment usage
        usage["requests_minute"] += amount
        usage["requests_day"] += amount
        usage["requests_month"] += amount

        # Persist usage (simplified)
        self._persist_usage(tenant_id, usage_type, amount, metadata)

    def get_tenant_usage(
        self,
        tenant_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict:
        """
        Get usage statistics for tenant.

        Args:
            tenant_id: Tenant ID
            start_date: Start date for query
            end_date: End date for query

        Returns:
            Usage statistics
        """
        # Simplified - real implementation would query database
        usage = self.usage_cache.get(tenant_id, {})

        return {
            "tenant_id": tenant_id,
            "current_minute": usage.get("requests_minute", 0),
            "current_day": usage.get("requests_day", 0),
            "current_month": usage.get("requests_month", 0),
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
        }

    def list_tenants(
        self,
        tier: Optional[TenantTier] = None,
        active_only: bool = True,
    ) -> List[Tenant]:
        """
        List tenants with optional filtering.

        Args:
            tier: Filter by tier
            active_only: Only return active tenants

        Returns:
            List of tenants
        """
        tenants = list(self.tenants.values())

        if tier:
            tenants = [t for t in tenants if t.tier == tier]

        if active_only:
            tenants = [t for t in tenants if t.is_active]

        return tenants

    def deactivate_tenant(self, tenant_id: str) -> bool:
        """Deactivate a tenant."""
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return False

        tenant.is_active = False
        self._persist_tenant(tenant)

        logger.info(f"Deactivated tenant: {tenant_id}")
        return True

    def activate_tenant(self, tenant_id: str) -> bool:
        """Activate a tenant."""
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return False

        tenant.is_active = True
        self._persist_tenant(tenant)

        logger.info(f"Activated tenant: {tenant_id}")
        return True

    def _get_current_usage(self, tenant_id: str, quota_type: str) -> int:
        """Get current usage for quota type."""
        usage = self.usage_cache.get(tenant_id, {})

        # Map quota types to usage keys
        mapping = {
            "requests_per_minute": "requests_minute",
            "requests_per_day": "requests_day",
            "requests_per_month": "requests_month",
        }

        usage_key = mapping.get(quota_type, "")
        return usage.get(usage_key, 0)

    def _persist_tenant(self, tenant: Tenant) -> None:
        """Persist tenant to storage."""
        # Simplified - real implementation would write to database
        pass

    def _persist_usage(
        self, tenant_id: str, usage_type: str, amount: int, metadata: Optional[Dict]
    ) -> None:
        """Persist usage to storage."""
        # Simplified - real implementation would write to time-series database
        pass


class DataIsolation:
    """
    Ensures data isolation between tenants.

    Provides:
    - Tenant-scoped database queries
    - File storage isolation
    - Cache isolation
    """

    @staticmethod
    def get_tenant_storage_path(tenant_id: str, base_path: str = "/data") -> str:
        """Get isolated storage path for tenant."""
        return f"{base_path}/tenants/{tenant_id}"

    @staticmethod
    def get_tenant_cache_key(tenant_id: str, key: str) -> str:
        """Get isolated cache key for tenant."""
        return f"tenant:{tenant_id}:{key}"

    @staticmethod
    def validate_tenant_access(tenant_id: str, resource_id: str) -> bool:
        """Validate tenant has access to resource."""
        # Check if resource belongs to tenant
        # Simplified - real implementation would query database
        return True


class WhiteLabelConfig:
    """White-label configuration for tenants."""

    @dataclass
    class BrandingConfig:
        """Branding configuration."""
        logo_url: Optional[str] = None
        primary_color: str = "#007bff"
        secondary_color: str = "#6c757d"
        company_name: str = ""
        support_email: str = ""
        support_url: str = ""
        terms_url: str = ""
        privacy_url: str = ""

    @staticmethod
    def get_branding(tenant: Tenant) -> BrandingConfig:
        """Get branding configuration for tenant."""
        config = tenant.white_label_config

        return WhiteLabelConfig.BrandingConfig(
            logo_url=config.get("logo_url"),
            primary_color=config.get("primary_color", "#007bff"),
            secondary_color=config.get("secondary_color", "#6c757d"),
            company_name=config.get("company_name", tenant.company),
            support_email=config.get("support_email", tenant.admin_email),
            support_url=config.get("support_url"),
            terms_url=config.get("terms_url"),
            privacy_url=config.get("privacy_url"),
        )
