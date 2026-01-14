"""
Domain Loader

Load research domain list from JSON file.
"""

import logging
from typing import List

from ..utils.file_utils import load_json

logger = logging.getLogger(__name__)


class DomainLoader:
    """Load research domains from file."""

    def __init__(self, domains_file: str) -> None:
        """
        Initialize the domain loader.

        Args:
            domains_file: Path to the domain list JSON file
        """
        self.domains_file = domains_file

    def load(self) -> List[str]:
        """
        Load domain list.

        Returns:
            List of domain names

        Raises:
            FileNotFoundError: File does not exist
            Exception: Load failed
        """
        logger.info(f"Loading research domains: {self.domains_file}")

        try:
            domains = load_json(self.domains_file)

            if not isinstance(domains, list):
                raise ValueError("domains.json should contain a list of strings")

            logger.info(f"Successfully loaded {len(domains)} research domains:")
            for i, domain in enumerate(domains, 1):
                logger.info(f"  {i}. {domain}")

            return domains

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Domain file not found: {self.domains_file}\n"
                "Please create this file with a JSON array of domain names"
            )
        except Exception as e:
            raise Exception(f"Failed to load domains: {e}")


if __name__ == "__main__":
    import os
    import tempfile
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print("=" * 60)
    print("Domain Loader Module Test")
    print("=" * 60)

    # Create temporary test file
    temp_dir = tempfile.mkdtemp()
    test_file = os.path.join(temp_dir, "test_domains.json")

    try:
        # Test normal loading
        print("\n1. Testing normal loading...")
        test_domains = [
            "Artificial Intelligence",
            "Healthcare",
            "Finance",
            "Education",
            "Climate Science",
        ]
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(test_domains, f)

        loader = DomainLoader(test_file)
        domains = loader.load()
        assert domains == test_domains
        print(f"   ✓ Successfully loaded {len(domains)} domains")

        # Test file not found
        print("\n2. Testing file not found...")
        loader_missing = DomainLoader("/not/exist/path.json")
        try:
            loader_missing.load()
            print("   ✗ Should have raised exception")
        except FileNotFoundError as e:
            print(f"   ✓ Correctly raised FileNotFoundError")

        # Test empty list
        print("\n3. Testing empty list...")
        empty_file = os.path.join(temp_dir, "empty.json")
        with open(empty_file, "w") as f:
            json.dump([], f)

        loader_empty = DomainLoader(empty_file)
        domains = loader_empty.load()
        assert domains == []
        print("   ✓ Correctly handled empty list")

    finally:
        import shutil

        shutil.rmtree(temp_dir)

    print("\n" + "=" * 60)
    print("Domain Loader module test complete!")
