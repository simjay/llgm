"""Explicit input conversion to integer Unix milliseconds without guessing source time."""

from __future__ import annotations

from datetime import date, datetime, time, timezone, tzinfo
from typing import Any, Mapping

from llgm.core.errors import SchemaError


def validate_instant_ms(value: int | None, name: str = "instant_ms") -> int | None:
    """Accept an integer Unix-millisecond instant or unknown, excluding booleans."""
    if value is not None and type(value) is not int:
        raise SchemaError(f"{name} must be integer Unix milliseconds or None")
    return value


def parse_instant_ms(value: str, *, date_only_timezone: tzinfo | None = None) -> int:
    """Parse a declared ISO instant. Date-only input requires an explicit timezone.

    Date-only conversion means the start of that civil day in the supplied zone.
    Preserve the original text separately when it expresses imprecise source time.
    Naive timestamps and fractional precision finer than milliseconds are rejected.
    """
    if not isinstance(value, str):
        raise SchemaError("An ISO instant must be text")
    try:
        if len(value) == 10:
            if date_only_timezone is None:
                raise ValueError("date-only input requires an explicit timezone")
            instant = datetime.combine(date.fromisoformat(value), time(), date_only_timezone)
        else:
            instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if instant.tzinfo is None or instant.utcoffset() is None:
            raise ValueError("an explicit timezone is required")
        if instant.microsecond % 1000:
            raise ValueError("instant precision is finer than milliseconds")
        delta = instant.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
        return delta.days * 86400000 + delta.seconds * 1000 + delta.microseconds // 1000
    except (ValueError, OverflowError, TypeError) as error:
        raise SchemaError(f"Invalid ISO instant: {error}") from error


def legacy_iso_to_ms(value: str) -> int:
    """Convert historical ISO validity input with its declared UTC date-boundary rule."""
    return parse_instant_ms(value, date_only_timezone=timezone.utc)


def normalize_applicability(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize explicit legacy ISO bounds while retaining unknown selectors for review."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SchemaError("Applicability must be a mapping")
    normalized = dict(value)
    for old, new in (("valid_from", "valid_from_ms"), ("valid_until", "valid_until_ms")):
        if old in normalized:
            if new in normalized:
                raise SchemaError(f"Specify either {old} or {new}, not both")
            normalized[new] = legacy_iso_to_ms(normalized.pop(old))
        if new in normalized:
            if normalized[new] is None:
                raise SchemaError(f"Omit an unknown {new} instead of storing null")
            validate_instant_ms(normalized[new], new)
    start, end = normalized.get("valid_from_ms"), normalized.get("valid_until_ms")
    if start is not None and end is not None and start >= end:
        raise SchemaError("valid_from_ms must precede valid_until_ms")
    return normalized


def match_applicability(
    applicability: Mapping[str, Any] | None,
    *,
    scope: Mapping[str, Any],
    as_of_ms: int | None,
) -> tuple[str, str]:
    """Classify declared scope and validity as active, inactive, or unresolved.

    Missing query values required by a rule, and unknown rules, remain unresolved.
    A mismatched scope excludes the record regardless of supplied query time.
    """
    rules = normalize_applicability(applicability)
    if set(rules) - {"scope", "valid_from_ms", "valid_until_ms"}:
        return "unresolved", "unknown applicability selector"
    required_scope = rules.get("scope", {})
    if not isinstance(required_scope, Mapping):
        raise SchemaError("Applicability scope must be a mapping")
    if any(key in scope and scope[key] != value for key, value in required_scope.items()):
        return "inactive", "scope mismatch"
    status, reason = "active", "explicit applicability matches"
    if any(key not in scope for key in required_scope):
        status, reason = "unresolved", "query scope missing"
    start, end = rules.get("valid_from_ms"), rules.get("valid_until_ms")
    if start is not None or end is not None:
        if as_of_ms is None:
            return "unresolved", "query valid time missing"
        if (start is not None and as_of_ms < start) or (end is not None and as_of_ms >= end):
            return "inactive", "outside valid-time interval"
    return status, reason
