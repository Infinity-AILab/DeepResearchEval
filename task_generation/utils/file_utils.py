"""
File IO Utilities

Provides JSON/JSONL file read/write functionality with incremental saving support.
"""

import json
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar

import jsonlines

logger = logging.getLogger(__name__)

T = TypeVar("T")


def ensure_dir(file_path: str) -> None:
    """
    Ensure the directory for a file exists, creating it if necessary.

    Args:
        file_path: Path to the file
    """
    dir_path = os.path.dirname(file_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)


def load_json(file_path: str) -> Any:
    """
    Load a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Parsed Python object

    Raises:
        FileNotFoundError: File does not exist
        json.JSONDecodeError: JSON parse failed
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, file_path: str, indent: int = 2) -> None:
    """
    Save data to a JSON file.

    Args:
        data: Data to save
        file_path: Target file path
        indent: Indentation spaces
    """
    ensure_dir(file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    logger.debug(f"Saved JSON to: {file_path}")


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save data to a JSONL file.

    Args:
        data: List of dictionaries
        file_path: Target file path
    """
    ensure_dir(file_path)
    with jsonlines.open(file_path, "w") as writer:
        writer.write_all(data)
    logger.debug(f"Saved JSONL to: {file_path}")


def load_if_exists(
    file_path: str,
    default: T = None,
) -> T | Any:
    """
    Load file if it exists, otherwise return default value.

    Args:
        file_path: Path to the JSON file
        default: Default return value if file doesn't exist

    Returns:
        Loaded data or default value
    """
    if os.path.exists(file_path):
        try:
            data = load_json(file_path)
            logger.info(f"Loaded from cache: {file_path}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache {file_path}: {e}")
            return default
    return default


if __name__ == "__main__":
    import tempfile
    import shutil

    print("=" * 60)
    print("File Utilities Module Test")
    print("=" * 60)

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"\nUsing temporary directory: {temp_dir}")

    try:
        # Test ensure_dir
        print("\n1. Testing ensure_dir...")
        nested_path = os.path.join(temp_dir, "a", "b", "c", "file.json")
        ensure_dir(nested_path)
        assert os.path.isdir(os.path.dirname(nested_path))
        print("   ✓ Nested directory created successfully")

        # Test save_json and load_json
        print("\n2. Testing save_json / load_json...")
        test_data = {
            "name": "test_data",
            "items": [1, 2, 3],
            "nested": {"key": "value"},
        }
        json_path = os.path.join(temp_dir, "test.json")
        save_json(test_data, json_path)
        loaded = load_json(json_path)
        assert loaded == test_data
        print("   ✓ JSON save and load correct")

        # Test save_jsonl
        print("\n3. Testing save_jsonl...")
        jsonl_data = [
            {"id": 1, "text": "first"},
            {"id": 2, "text": "second"},
            {"id": 3, "text": "third"},
        ]
        jsonl_path = os.path.join(temp_dir, "test.jsonl")
        save_jsonl(jsonl_data, jsonl_path)
        # Verify file content
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 3
        print("   ✓ JSONL saved correctly")

        # Test load_if_exists
        print("\n4. Testing load_if_exists...")
        # Existing file
        result = load_if_exists(json_path, default=[])
        assert result == test_data
        print("   ✓ Load existing file")

        # Non-existing file
        result = load_if_exists(
            os.path.join(temp_dir, "not_exist.json"),
            default={"default": True},
        )
        assert result == {"default": True}
        print("   ✓ Returns default when file doesn't exist")

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temporary directory: {temp_dir}")

    print("\n" + "=" * 60)
    print("File Utilities module test complete!")
