"""
LLM API Client

Supports OpenAI and OpenRouter API types with exponential backoff retry mechanism.
"""

import os
import time
import logging
from typing import List, Dict, Literal

from openai import OpenAI

logger = logging.getLogger(__name__)


class APIClient:
    """LLM API client wrapping OpenAI SDK calls."""

    ERROR_SENTINEL = "$ERROR$"

    def __init__(
        self,
        model_name: str = "gpt-5-mini",
        api_type: Literal["openai", "openrouter"] = "openai",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """
        Initialize the API client.

        Args:
            model_name: Name of the model to use
            api_type: API type, supports "openai" or "openrouter"
            api_key: API key, reads from environment variable if None
            base_url: API base URL, uses default if None
        """
        self.model_name = model_name
        self.api_type = api_type

        if api_type == "openai":
            self.client = OpenAI(
                api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
                base_url=base_url or "https://api.openai.com/v1",
            )
        elif api_type == "openrouter":
            self.client = OpenAI(
                api_key=api_key or os.environ.get("OPENROUTER_KEY", ""),
                base_url=base_url or "https://openrouter.ai/api/v1",
            )
        else:
            raise ValueError(f"Unsupported API type: {api_type}")

        logger.info(f"Initialized API client: model={model_name}, type={api_type}")

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 8192,
        retry_count: int = 3,
    ) -> str:
        """
        Call LLM to generate response.

        Args:
            messages: List of messages in OpenAI format
            max_tokens: Maximum tokens to generate
            retry_count: Number of retries on failure

        Returns:
            Generated response text, returns ERROR_SENTINEL on failure
        """
        for attempt in range(retry_count):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                )
                response = completion.choices[0].message.content
                return response or ""

            except Exception as e:
                wait_time = 2**attempt
                logger.warning(
                    f"API call failed (attempt {attempt + 1}/{retry_count}): {e}"
                )
                if attempt == retry_count - 1:
                    logger.error(f"API call finally failed: {e}")
                    return self.ERROR_SENTINEL
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)

        return self.ERROR_SENTINEL

    def is_error(self, response: str) -> bool:
        """Check if response is an error sentinel."""
        return response == self.ERROR_SENTINEL


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print("=" * 60)
    print("API Client Module Test")
    print("=" * 60)

    # Test client initialization
    print("\n1. Testing client initialization...")
    try:
        client = APIClient(
            model_name="gpt-5-mini",
            api_type="openai",
            api_key="test-key",  # Using test key
        )
        print("   ✓ OpenAI type client initialized successfully")
    except Exception as e:
        print(f"   ✗ Initialization failed: {e}")

    try:
        client = APIClient(
            model_name="anthropic/claude-3-opus",
            api_type="openrouter",
            api_key="test-key",
        )
        print("   ✓ OpenRouter type client initialized successfully")
    except Exception as e:
        print(f"   ✗ Initialization failed: {e}")

    # Test error checking
    print("\n2. Testing error checking...")
    assert client.is_error(APIClient.ERROR_SENTINEL) is True
    assert client.is_error("normal response") is False
    print("   ✓ Error checking logic correct")

    # Simulate API call structure (not actually calling)
    print("\n3. Simulating API call structure...")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    print(f"   Message format: {messages}")
    print("   ✓ Message format correct")

    print("\n" + "=" * 60)
    print("API Client module test complete!")
    print("Note: Actual API calls require a valid API Key")
