"""JSON output types and a shallow guard for untrusted input dictionaries."""

from typing import TypeGuard

type JSONScalar = None | bool | int | float | str
type JSONValue = JSONScalar | list[JSONValue] | dict[str, JSONValue]
type JSONObject = dict[str, JSONValue]


def is_string_keyed_dict(value: object) -> TypeGuard[dict[str, object]]:
    """Narrow a dictionary without trusting its values or traversing unknown fields."""
    return isinstance(value, dict) and all(isinstance(key, str) for key in value)
