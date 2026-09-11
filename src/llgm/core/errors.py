"""Public errors for storage, evidence, and inference failures."""


class LLGMError(Exception):
    """Base class for expected library failures."""


class ConfigurationError(LLGMError, ValueError):
    """Invalid or incompatible configuration."""


class CapabilityError(ConfigurationError):
    """The selected adapter does not support a required capability."""


class ConflictError(LLGMError):
    """An idempotency or optimistic concurrency precondition failed."""


class ReferenceResolutionError(LLGMError):
    """An evidence reference names missing data or an invalid text range."""


class BudgetExceeded(LLGMError):
    """An execution allowance or deadline was exhausted."""


class ProviderError(LLGMError):
    """A remote model failed, refused, or returned an incomplete response."""


class SchemaError(LLGMError, ValueError):
    """A stored record or model operation violates its data contract."""
