"""JSON boundaries shared by inference protocols and the Docker transport."""

import json

from llgm.core.errors import ConfigurationError, SchemaError


def _unique_object(pairs):
    """Reject duplicate fields rather than choosing a model operation implicitly."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def parse_object(text, error_message):
    """Parse one JSON object, rejecting duplicate keys, trailing data, and non-text input."""
    try:
        if not isinstance(text, str):
            raise ValueError("JSON response must be text")
        result = json.loads(text, object_pairs_hook=_unique_object)
        if not isinstance(result, dict):
            raise ValueError("JSON response must be an object")
    except (ValueError, RecursionError) as error:
        raise SchemaError(error_message) from error
    return result


def validate_json_types(value):
    """Reject non-JSON values and non-string keys before copying external context.

    Callers serialize first to reject cycles and nonfinite numbers before this
    recursive check. The JSON encoder alone would coerce tuples and object keys.
    """
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ConfigurationError("Context object keys must be strings")
        for child in value.values():
            validate_json_types(child)
    elif isinstance(value, list):
        for child in value:
            validate_json_types(child)
    elif value is not None and not isinstance(value, (str, bool, int, float)):
        raise ConfigurationError("Context accepts JSON types only")
