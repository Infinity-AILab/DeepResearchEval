"""
JSON Parsing Utilities

Extract and parse JSON content from LLM responses.
"""

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def extract_json(text: str) -> Optional[str]:
    """
    Extract JSON string from text.

    Finds content between the first '{' and last '}'.

    Args:
        text: Text that may contain JSON

    Returns:
        Extracted JSON string, or None if not found
    """
    json_start = text.find("{")
    json_end = text.rfind("}") + 1

    if json_start >= 0 and json_end > json_start:
        return text[json_start:json_end]
    return None


def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Safely parse JSON from text.

    First extracts JSON string, then attempts to parse as dictionary.

    Args:
        text: Text that may contain JSON

    Returns:
        Parsed dictionary, or None on parse failure
    """
    json_str = extract_json(text)
    if json_str is None:
        logger.warning("No JSON content found in response")
        return None

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse failed: {e}")
        logger.debug(f"Original JSON string: {json_str[:200]}...")
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("JSON Utilities Module Test")
    print("=" * 60)

    # Test extract_json
    print("\n1. Testing extract_json...")

    # Normal case
    text1 = 'Here is the result: {"name": "Alice", "age": 30}'
    result = extract_json(text1)
    assert result == '{"name": "Alice", "age": 30}'
    print("   ✓ Normal JSON extraction")

    # With surrounding text
    text2 = """
    Based on my analysis, here is the JSON:
    {
        "status": "success",
        "data": [1, 2, 3]
    }
    Hope this helps!
    """
    result = extract_json(text2)
    assert '"status": "success"' in result
    print("   ✓ Extract JSON from multiline text")

    # No JSON
    text3 = "No JSON here"
    result = extract_json(text3)
    assert result is None
    print("   ✓ Returns None when no JSON")

    # Test safe_parse_json
    print("\n2. Testing safe_parse_json...")

    # Normal parsing
    text4 = 'Response: {"key": "value", "number": 42}'
    result = safe_parse_json(text4)
    assert result == {"key": "value", "number": 42}
    print("   ✓ Normal JSON to dict parsing")

    # Nested JSON
    text5 = '{"outer": {"inner": "nested"}, "list": [1, 2, 3]}'
    result = safe_parse_json(text5)
    assert result["outer"]["inner"] == "nested"
    print("   ✓ Parse nested JSON")

    # Invalid JSON
    text6 = "{invalid json content}"
    result = safe_parse_json(text6)
    assert result is None
    print("   ✓ Returns None for invalid JSON")

    print("\n" + "=" * 60)
    print("JSON Utilities module test complete!")
