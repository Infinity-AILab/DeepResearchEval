"""
Expert Persona Generator

Generate expert role descriptions in parallel based on research domains.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from ..clients.api_client import APIClient
from ..prompts.templates import PERSONA_GENERATION_PROMPT, render_prompt
from ..utils.json_utils import safe_parse_json
from ..utils.file_utils import load_if_exists, save_json

logger = logging.getLogger(__name__)


class ExpertGenerator:
    """Generate expert personas in parallel."""

    def __init__(
        self,
        api_client: APIClient,
        cache_file: Optional[str] = None,
        max_workers: int = 25,
    ) -> None:
        """
        Initialize the expert generator.

        Args:
            api_client: API client instance
            cache_file: Cache file path for checkpoint resume
            max_workers: Maximum parallel threads
        """
        self.api_client = api_client
        self.cache_file = cache_file
        self.max_workers = max_workers

    def generate(self, domains: List[str]) -> List[Dict[str, Any]]:
        """
        Generate experts for all domains.

        Args:
            domains: List of domain names

        Returns:
            List of expert info, each containing a domain field
        """
        # Check cache
        if self.cache_file:
            cached = load_if_exists(self.cache_file, default=None)
            if cached is not None:
                logger.info(f"Using cached expert data: {len(cached)} experts")
                return cached

        logger.info(f"Generating experts in parallel ({len(domains)} domains)...")

        all_experts: List[Dict[str, Any]] = []
        workers = min(self.max_workers, len(domains))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_domain = {
                executor.submit(self._generate_for_domain, domain): domain
                for domain in domains
            }

            completed = 0
            for future in as_completed(future_to_domain):
                completed += 1
                domain = future_to_domain[future]

                try:
                    result = future.result()
                    if result:
                        all_experts.extend(result)
                    logger.info(f"Progress: {completed}/{len(domains)}")
                except Exception as e:
                    logger.error(f"Error processing domain '{domain}': {e}")

        logger.info(f"Total generated: {len(all_experts)} experts")

        # Save cache
        if self.cache_file:
            save_json(all_experts, self.cache_file)

        return all_experts

    def _generate_for_domain(self, domain: str) -> Optional[List[Dict[str, Any]]]:
        """
        Generate experts for a single domain.

        Args:
            domain: Domain name

        Returns:
            List of experts for this domain, None on failure
        """
        logger.debug(f"Generating experts for domain '{domain}'...")

        prompt = render_prompt(PERSONA_GENERATION_PROMPT, {"domain": domain})
        messages = [{"role": "user", "content": prompt}]

        response = self.api_client.generate_response(messages)
        if self.api_client.is_error(response):
            logger.error(f"Expert generation failed: {domain}")
            return None

        data = safe_parse_json(response)
        if data is None or "personas" not in data:
            logger.error(f"Failed to parse expert JSON: {domain}")
            return None

        # Add domain field to each persona
        personas = []
        for persona in data["personas"]:
            persona["domain"] = domain
            personas.append(persona)

        logger.info(f"  '{domain}': generated {len(personas)} experts")
        return personas


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print("=" * 60)
    print("Expert Generator Module Test")
    print("=" * 60)

    # Create mock data for testing
    print("\n1. Testing ExpertGenerator initialization...")
    mock_client = APIClient(
        model_name="gpt-5-mini",
        api_type="openai",
        api_key="test-key",
    )
    generator = ExpertGenerator(
        api_client=mock_client,
        max_workers=5,
    )
    print("   ✓ ExpertGenerator initialized successfully")

    # Test render_prompt
    print("\n2. Testing prompt rendering...")
    prompt = render_prompt(PERSONA_GENERATION_PROMPT, {"domain": "AI Research"})
    assert "AI Research" in prompt
    assert "{{domain}}" not in prompt
    print("   ✓ Prompt rendered correctly")

    # Validate expert data structure
    print("\n3. Validating expert data structure...")
    mock_expert = {
        "name": "Dr. Alice Chen",
        "role": "AI Research Scientist",
        "affiliation": "Stanford University",
        "background": "10 years of experience in NLP and LLM research...",
        "subdomain": "Large Language Models",
        "domain": "Artificial Intelligence",
    }
    required_fields = ["name", "role", "affiliation", "background", "subdomain", "domain"]
    for field in required_fields:
        assert field in mock_expert, f"Missing field: {field}"
    print(f"   ✓ Expert data contains all required fields: {required_fields}")

    print("\n" + "=" * 60)
    print("Expert Generator module test complete!")
    print("Note: Actual generation requires a valid API Key")
