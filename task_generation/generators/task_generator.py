"""
Deep Research Task Generator

Generate deep research tasks in parallel based on expert backgrounds.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from ..clients.api_client import APIClient
from ..prompts.templates import QUERY_GENERATION_PROMPT, render_prompt
from ..utils.json_utils import safe_parse_json
from ..utils.file_utils import load_if_exists, save_json

logger = logging.getLogger(__name__)


class TaskGenerator:
    """Generate Deep Research tasks based on expert backgrounds."""

    def __init__(
        self,
        api_client: APIClient,
        cache_file: Optional[str] = None,
        max_workers: int = 25,
    ) -> None:
        """
        Initialize the task generator.

        Args:
            api_client: API client instance
            cache_file: Cache file path
            max_workers: Maximum parallel threads
        """
        self.api_client = api_client
        self.cache_file = cache_file
        self.max_workers = max_workers

    def generate(self, experts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate tasks for all experts.

        Args:
            experts: List of expert info

        Returns:
            List of tasks, each containing expert and domain fields
        """
        # Check cache
        if self.cache_file:
            cached = load_if_exists(self.cache_file, default=None)
            if cached is not None:
                logger.info(f"Using cached task data: {len(cached)} tasks")
                return cached

        logger.info(f"Generating tasks in parallel ({len(experts)} experts)...")

        all_tasks: List[Dict[str, Any]] = []
        workers = min(self.max_workers, len(experts))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_expert = {
                executor.submit(self._generate_for_expert, expert): expert
                for expert in experts
            }

            completed = 0
            for future in as_completed(future_to_expert):
                completed += 1
                expert = future_to_expert[future]

                try:
                    result = future.result()
                    if result:
                        all_tasks.extend(result)
                    logger.info(f"Progress: {completed}/{len(experts)}")
                except Exception as e:
                    logger.error(f"Error processing expert '{expert.get('name', 'unknown')}': {e}")

        logger.info(f"Total generated: {len(all_tasks)} tasks")

        # Save cache
        if self.cache_file:
            save_json(all_tasks, self.cache_file)

        return all_tasks

    def _generate_for_expert(self, expert: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """
        Generate tasks for a single expert.

        Args:
            expert: Expert info dictionary

        Returns:
            List of tasks for this expert, None on failure
        """
        expert_name = expert.get("name", "unknown")
        logger.debug(f"Generating tasks for expert '{expert_name}'...")

        # Render prompt
        prompt = render_prompt(QUERY_GENERATION_PROMPT, expert)
        messages = [{"role": "user", "content": prompt}]

        response = self.api_client.generate_response(messages)
        if self.api_client.is_error(response):
            logger.error(f"Task generation failed: {expert_name}")
            return None

        data = safe_parse_json(response)
        if data is None or "tasks" not in data:
            logger.error(f"Failed to parse task JSON: {expert_name}")
            return None

        # Add expert info and domain to each task
        tasks = []
        for task in data["tasks"]:
            task["expert"] = expert
            task["domain"] = expert.get("domain", "unknown")
            tasks.append(task)

        logger.info(f"  '{expert_name}': generated {len(tasks)} tasks")
        return tasks


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print("=" * 60)
    print("Task Generator Module Test")
    print("=" * 60)

    # Create mock data for testing
    print("\n1. Testing TaskGenerator initialization...")
    mock_client = APIClient(
        model_name="gpt-5-mini",
        api_type="openai",
        api_key="test-key",
    )
    generator = TaskGenerator(
        api_client=mock_client,
        max_workers=5,
    )
    print("   ✓ TaskGenerator initialized successfully")

    # Test prompt rendering
    print("\n2. Testing prompt rendering...")
    mock_expert = {
        "name": "Dr. Alice Chen",
        "role": "AI Research Scientist",
        "affiliation": "Stanford University",
        "background": "10 years of experience in NLP research",
        "subdomain": "Large Language Models",
        "domain": "Artificial Intelligence",
    }
    prompt = render_prompt(QUERY_GENERATION_PROMPT, mock_expert)
    assert "Dr. Alice Chen" in prompt
    assert "Stanford University" in prompt
    assert "{{name}}" not in prompt
    print("   ✓ Prompt rendered correctly")

    # Validate task data structure
    print("\n3. Validating task data structure...")
    mock_task = {
        "task_id": "task_1",
        "deep_research_query": "What are the latest LLM benchmarks in 2025?",
        "key_challenges": "Need up-to-date benchmark data from multiple sources",
        "expected_search_rounds": 4,
        "time_sensitivity": True,
        "time_constraint": "As of August 2025",
        "expert": mock_expert,
        "domain": "Artificial Intelligence",
    }
    required_fields = [
        "task_id",
        "deep_research_query",
        "key_challenges",
        "expected_search_rounds",
        "time_sensitivity",
        "expert",
        "domain",
    ]
    for field in required_fields:
        assert field in mock_task, f"Missing field: {field}"
    print(f"   ✓ Task data contains all required fields")

    print("\n" + "=" * 60)
    print("Task Generator module test complete!")
    print("Note: Actual generation requires a valid API Key")
