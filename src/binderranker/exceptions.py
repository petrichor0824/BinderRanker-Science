"""Domain-specific BinderRanker exceptions."""


class BinderRankerError(Exception):
    """Base error for user-facing BinderRanker failures."""


class PDBValidationError(BinderRankerError):
    """Raised when strict PDB validation finds ambiguous or invalid input."""


class TargetPoolValidationError(BinderRankerError):
    """Raised when a candidate pool does not represent one compatible target."""


class OutputExistsError(BinderRankerError):
    """Raised when an output would be overwritten without explicit permission."""

