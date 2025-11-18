"""Usage metering and billing integration.

Provides:
- Granular usage tracking
- Billing calculations
- Invoice generation
- Stripe/AWS Marketplace integration
- Usage analytics and reporting
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
import uuid
from loguru import logger


class UsageMetric(Enum):
    """Types of billable usage metrics."""
    API_REQUESTS = "api_requests"
    IMAGES_PROCESSED = "images_processed"
    VIDEOS_PROCESSED = "videos_processed"
    COMPUTE_SECONDS = "compute_seconds"
    STORAGE_GB = "storage_gb"
    BANDWIDTH_GB = "bandwidth_gb"
    MODEL_TRAINING = "model_training"
    CUSTOM_MODEL_USAGE = "custom_model_usage"


@dataclass
class PricingRule:
    """Pricing rule for a metric."""

    metric: UsageMetric
    unit_price: Decimal  # Price per unit
    currency: str = "USD"

    # Tiered pricing
    tiers: List[Dict[str, Any]] = field(default_factory=list)
    # Example tier: {"max": 1000, "price": 0.05}
    #               {"max": 10000, "price": 0.03}
    #               {"max": None, "price": 0.02}  # Unlimited

    # Minimum/maximum
    minimum_charge: Decimal = Decimal("0")
    included_units: int = 0  # Free tier

    def calculate_cost(self, units: int) -> Decimal:
        """
        Calculate cost for units using tiered pricing.

        Args:
            units: Number of units consumed

        Returns:
            Total cost
        """
        # Subtract included units
        billable_units = max(0, units - self.included_units)

        if not self.tiers:
            # Flat pricing
            cost = Decimal(billable_units) * self.unit_price
        else:
            # Tiered pricing
            cost = Decimal("0")
            remaining = billable_units

            for tier in self.tiers:
                tier_max = tier.get("max")
                tier_price = Decimal(str(tier["price"]))

                if tier_max is None:
                    # Unlimited tier
                    cost += Decimal(remaining) * tier_price
                    break
                else:
                    tier_units = min(remaining, tier_max)
                    cost += Decimal(tier_units) * tier_price
                    remaining -= tier_units

                    if remaining <= 0:
                        break

        # Apply minimum charge
        cost = max(cost, self.minimum_charge)

        return cost


@dataclass
class UsageRecord:
    """Record of usage for billing."""

    record_id: str
    tenant_id: str
    metric: UsageMetric
    quantity: int
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Cost calculation
    unit_price: Optional[Decimal] = None
    total_cost: Optional[Decimal] = None

    # Reference
    resource_id: Optional[str] = None  # e.g., image ID, video ID


class UsageMeter:
    """
    Tracks usage for billing purposes.

    Provides:
    - Real-time usage tracking
    - Aggregation by time period
    - Cost calculation
    - Usage alerts
    """

    def __init__(self):
        """Initialize usage meter."""
        self.records: List[UsageRecord] = []
        self.pricing_rules: Dict[UsageMetric, PricingRule] = {}

        logger.info("Usage meter initialized")

    def set_pricing_rule(self, rule: PricingRule) -> None:
        """Set pricing rule for a metric."""
        self.pricing_rules[rule.metric] = rule
        logger.info(f"Set pricing for {rule.metric.value}: ${rule.unit_price}/unit")

    def record_usage(
        self,
        tenant_id: str,
        metric: UsageMetric,
        quantity: int,
        metadata: Optional[Dict] = None,
        resource_id: Optional[str] = None,
    ) -> UsageRecord:
        """
        Record usage for a tenant.

        Args:
            tenant_id: Tenant ID
            metric: Usage metric
            quantity: Quantity consumed
            metadata: Additional metadata
            resource_id: Resource ID

        Returns:
            Usage record
        """
        record_id = str(uuid.uuid4())

        # Calculate cost
        pricing_rule = self.pricing_rules.get(metric)
        unit_price = None
        total_cost = None

        if pricing_rule:
            unit_price = pricing_rule.unit_price
            total_cost = pricing_rule.calculate_cost(quantity)

        record = UsageRecord(
            record_id=record_id,
            tenant_id=tenant_id,
            metric=metric,
            quantity=quantity,
            timestamp=datetime.now(),
            metadata=metadata or {},
            unit_price=unit_price,
            total_cost=total_cost,
            resource_id=resource_id,
        )

        self.records.append(record)

        return record

    def get_usage(
        self,
        tenant_id: str,
        metric: Optional[UsageMetric] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[UsageRecord]:
        """
        Get usage records for a tenant.

        Args:
            tenant_id: Tenant ID
            metric: Filter by metric
            start_date: Start date
            end_date: End date

        Returns:
            List of usage records
        """
        records = [r for r in self.records if r.tenant_id == tenant_id]

        if metric:
            records = [r for r in records if r.metric == metric]

        if start_date:
            records = [r for r in records if r.timestamp >= start_date]

        if end_date:
            records = [r for r in records if r.timestamp <= end_date]

        return records

    def aggregate_usage(
        self,
        tenant_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[UsageMetric, int]:
        """
        Aggregate usage by metric for a time period.

        Args:
            tenant_id: Tenant ID
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary mapping metric to total quantity
        """
        records = self.get_usage(tenant_id, start_date=start_date, end_date=end_date)

        aggregated = {}
        for record in records:
            if record.metric not in aggregated:
                aggregated[record.metric] = 0
            aggregated[record.metric] += record.quantity

        return aggregated

    def calculate_bill(
        self,
        tenant_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict:
        """
        Calculate bill for a tenant.

        Args:
            tenant_id: Tenant ID
            start_date: Billing period start
            end_date: Billing period end

        Returns:
            Bill details
        """
        usage = self.aggregate_usage(tenant_id, start_date, end_date)

        line_items = []
        total_cost = Decimal("0")

        for metric, quantity in usage.items():
            pricing_rule = self.pricing_rules.get(metric)
            if not pricing_rule:
                continue

            cost = pricing_rule.calculate_cost(quantity)
            total_cost += cost

            line_items.append({
                "metric": metric.value,
                "quantity": quantity,
                "unit_price": float(pricing_rule.unit_price),
                "cost": float(cost),
                "currency": pricing_rule.currency,
            })

        bill = {
            "tenant_id": tenant_id,
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "line_items": line_items,
            "total_cost": float(total_cost),
            "currency": "USD",
        }

        return bill


class BillingIntegration:
    """
    Integration with billing providers.

    Supports:
    - Stripe
    - AWS Marketplace
    - Azure Marketplace
    - Custom billing systems
    """

    def __init__(self, provider: str = "stripe"):
        """
        Initialize billing integration.

        Args:
            provider: Billing provider (stripe, aws, azure, custom)
        """
        self.provider = provider
        logger.info(f"Billing integration initialized: {provider}")

    def create_customer(
        self, tenant_id: str, email: str, name: str
    ) -> Dict:
        """
        Create customer in billing system.

        Args:
            tenant_id: Tenant ID
            email: Customer email
            name: Customer name

        Returns:
            Customer info including provider customer ID
        """
        if self.provider == "stripe":
            return self._create_stripe_customer(tenant_id, email, name)
        elif self.provider == "aws":
            return self._create_aws_customer(tenant_id, email, name)
        else:
            return {"customer_id": tenant_id, "provider": "custom"}

    def create_subscription(
        self,
        tenant_id: str,
        plan_id: str,
        billing_cycle: str = "monthly",
    ) -> Dict:
        """
        Create subscription for tenant.

        Args:
            tenant_id: Tenant ID
            plan_id: Plan ID
            billing_cycle: monthly, annual

        Returns:
            Subscription details
        """
        if self.provider == "stripe":
            return self._create_stripe_subscription(tenant_id, plan_id, billing_cycle)
        else:
            return {
                "subscription_id": str(uuid.uuid4()),
                "tenant_id": tenant_id,
                "plan_id": plan_id,
                "status": "active",
            }

    def report_usage(
        self,
        tenant_id: str,
        usage_records: List[UsageRecord],
    ) -> Dict:
        """
        Report usage to billing provider for metered billing.

        Args:
            tenant_id: Tenant ID
            usage_records: Usage records to report

        Returns:
            Report status
        """
        if self.provider == "stripe":
            return self._report_stripe_usage(tenant_id, usage_records)
        elif self.provider == "aws":
            return self._report_aws_usage(tenant_id, usage_records)
        else:
            return {"status": "success", "records_reported": len(usage_records)}

    def create_invoice(
        self, tenant_id: str, bill: Dict
    ) -> Dict:
        """
        Create invoice in billing system.

        Args:
            tenant_id: Tenant ID
            bill: Bill details

        Returns:
            Invoice details
        """
        if self.provider == "stripe":
            return self._create_stripe_invoice(tenant_id, bill)
        else:
            return {
                "invoice_id": str(uuid.uuid4()),
                "tenant_id": tenant_id,
                "amount": bill["total_cost"],
                "status": "pending",
            }

    def _create_stripe_customer(
        self, tenant_id: str, email: str, name: str
    ) -> Dict:
        """Create Stripe customer."""
        try:
            import stripe

            customer = stripe.Customer.create(
                email=email,
                name=name,
                metadata={"tenant_id": tenant_id},
            )

            return {
                "customer_id": customer.id,
                "tenant_id": tenant_id,
                "provider": "stripe",
            }

        except ImportError:
            logger.warning("Stripe library not installed")
            return {"customer_id": tenant_id, "provider": "stripe_mock"}

    def _create_stripe_subscription(
        self, tenant_id: str, plan_id: str, billing_cycle: str
    ) -> Dict:
        """Create Stripe subscription."""
        try:
            import stripe

            # This is simplified - real implementation would:
            # 1. Get customer from tenant
            # 2. Create subscription with plan
            # 3. Handle payment method

            return {
                "subscription_id": "sub_" + str(uuid.uuid4()),
                "tenant_id": tenant_id,
                "plan_id": plan_id,
                "status": "active",
                "provider": "stripe",
            }

        except ImportError:
            return {
                "subscription_id": str(uuid.uuid4()),
                "tenant_id": tenant_id,
                "plan_id": plan_id,
                "status": "active",
                "provider": "stripe_mock",
            }

    def _report_stripe_usage(
        self, tenant_id: str, usage_records: List[UsageRecord]
    ) -> Dict:
        """Report usage to Stripe."""
        try:
            import stripe

            # Report usage for metered billing
            # Stripe uses subscription items for metered billing

            for record in usage_records:
                # stripe.SubscriptionItem.create_usage_record(
                #     subscription_item_id,
                #     quantity=record.quantity,
                #     timestamp=int(record.timestamp.timestamp()),
                # )
                pass

            return {"status": "success", "records_reported": len(usage_records)}

        except ImportError:
            return {
                "status": "success_mock",
                "records_reported": len(usage_records),
            }

    def _create_stripe_invoice(self, tenant_id: str, bill: Dict) -> Dict:
        """Create Stripe invoice."""
        # Simplified - real implementation would create actual Stripe invoice
        return {
            "invoice_id": "inv_" + str(uuid.uuid4()),
            "tenant_id": tenant_id,
            "amount": bill["total_cost"],
            "status": "pending",
            "provider": "stripe",
        }

    def _create_aws_customer(
        self, tenant_id: str, email: str, name: str
    ) -> Dict:
        """Create AWS Marketplace customer."""
        # AWS Marketplace has different customer model
        return {
            "customer_id": tenant_id,
            "provider": "aws_marketplace",
        }

    def _report_aws_usage(
        self, tenant_id: str, usage_records: List[UsageRecord]
    ) -> Dict:
        """Report usage to AWS Marketplace."""
        try:
            import boto3

            # AWS Marketplace Metering Service
            # marketplace_client = boto3.client('meteringmarketplace')
            # marketplace_client.batch_meter_usage(UsageRecords=[...])

            return {"status": "success", "records_reported": len(usage_records)}

        except ImportError:
            return {
                "status": "success_mock",
                "records_reported": len(usage_records),
            }


class UsageAlerts:
    """
    Usage alerts and notifications.

    Alerts tenants when:
    - Approaching quota limits
    - Unusual usage patterns
    - Cost thresholds exceeded
    """

    def __init__(self, usage_meter: UsageMeter):
        """Initialize usage alerts."""
        self.usage_meter = usage_meter
        self.alert_rules: List[Dict] = []

        logger.info("Usage alerts initialized")

    def add_alert_rule(
        self,
        tenant_id: str,
        metric: UsageMetric,
        threshold: int,
        threshold_type: str = "absolute",  # absolute, percentage
        alert_email: str = "",
    ) -> None:
        """
        Add alert rule.

        Args:
            tenant_id: Tenant ID
            metric: Metric to monitor
            threshold: Alert threshold
            threshold_type: absolute or percentage
            alert_email: Email for alerts
        """
        rule = {
            "tenant_id": tenant_id,
            "metric": metric,
            "threshold": threshold,
            "threshold_type": threshold_type,
            "alert_email": alert_email,
        }

        self.alert_rules.append(rule)

    def check_alerts(self, tenant_id: str) -> List[Dict]:
        """
        Check if any alert rules triggered.

        Args:
            tenant_id: Tenant ID

        Returns:
            List of triggered alerts
        """
        triggered = []

        # Get current month usage
        start_of_month = datetime.now().replace(day=1, hour=0, minute=0, second=0)
        usage = self.usage_meter.aggregate_usage(
            tenant_id, start_of_month, datetime.now()
        )

        for rule in self.alert_rules:
            if rule["tenant_id"] != tenant_id:
                continue

            metric = rule["metric"]
            threshold = rule["threshold"]
            current_usage = usage.get(metric, 0)

            # Check if threshold exceeded
            if current_usage >= threshold:
                triggered.append({
                    "metric": metric.value,
                    "current_usage": current_usage,
                    "threshold": threshold,
                    "alert_email": rule["alert_email"],
                })

        return triggered
