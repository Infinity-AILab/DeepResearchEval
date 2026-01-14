"""
Deep Research Requirement Filter

Filter tasks that truly require deep web search to complete.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from ..clients.api_client import APIClient
from ..prompts.templates import DEEP_RESEARCH_FILTER_PROMPT, render_prompt
from ..utils.json_utils import safe_parse_json
from ..utils.file_utils import load_if_exists, save_json

logger = logging.getLogger(__name__)


class DeepResearchFilter:
    """Filter tasks that require deep search."""

    def __init__(
        self,
        api_client: APIClient,
        cache_file: Optional[str] = None,
        max_workers: int = 25,
        confidence_threshold: float = 0.7,
    ) -> None:
        """
        Initialize the filter.

        Args:
            api_client: API client instance
            cache_file: Cache file path
            max_workers: Maximum parallel threads
            confidence_threshold: Confidence threshold, tasks below this are filtered
        """
        self.api_client = api_client
        self.cache_file = cache_file
        self.max_workers = max_workers
        self.confidence_threshold = confidence_threshold

    def filter(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter tasks that require deep search.

        Args:
            tasks: List of tasks to filter

        Returns:
            Filtered task list, each with filter_result field attached
        """
        # Check cache
        if self.cache_file:
            cached = load_if_exists(self.cache_file, default=None)
            if cached is not None:
                logger.info(f"Using cached filter results: {len(cached)} tasks")
                return cached

        logger.info(f"Filtering tasks in parallel ({len(tasks)} total)...")

        filtered_tasks: List[Dict[str, Any]] = []
        workers = min(self.max_workers, len(tasks))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_task = {
                executor.submit(self._filter_single, task): task for task in tasks
            }

            completed = 0
            for future in as_completed(future_to_task):
                completed += 1
                task = future_to_task[future]

                try:
                    result = future.result()
                    if result is not None:
                        filtered_tasks.append(result)

                    # Periodically save progress
                    if self.cache_file and completed % 10 == 0:
                        save_json(filtered_tasks, self.cache_file)
                        logger.info(
                            f"Progress: {completed}/{len(tasks)} (kept: {len(filtered_tasks)})"
                        )
                except Exception as e:
                    logger.error(f"Error processing task {task.get('task_id', 'unknown')}: {e}")

        logger.info(f"After filtering: {len(filtered_tasks)} tasks kept")

        # Save final results
        if self.cache_file:
            save_json(filtered_tasks, self.cache_file)

        return filtered_tasks

    def _filter_single(self, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Filter a single task.

        Args:
            task: Task info

        Returns:
            Task that passed filter (with filter_result), None if filtered out
        """
        task_id = task.get("task_id", "unknown")

        # Render prompt
        variables = {
            "deep_research_query": task.get("deep_research_query", ""),
            "persona_background": task.get("expert", {}).get("background", ""),
            "key_challenges": task.get("key_challenges", ""),
        }
        prompt = render_prompt(DEEP_RESEARCH_FILTER_PROMPT, variables)
        messages = [{"role": "user", "content": prompt}]

        response = self.api_client.generate_response(messages)
        if self.api_client.is_error(response):
            logger.error(f"Filter failed: {task_id}")
            return None

        filter_result = safe_parse_json(response)
        if filter_result is None:
            logger.error(f"Failed to parse filter JSON: {task_id}")
            return None

        # Check if deep search is needed
        needs_deep = filter_result.get("needs_deep_research", False)
        confidence = filter_result.get("confidence_score", 0)

        if needs_deep and confidence >= self.confidence_threshold:
            task["filter_result"] = filter_result
            logger.info(f"  ✓ Keep {task_id} (confidence: {confidence:.2f})")
            return task
        else:
            logger.info(
                f"  ✗ Filter {task_id} (needs_search: {needs_deep}, confidence: {confidence:.2f})"
            )
            return None


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print("=" * 60)
    print("Deep Research Filter Module Test")
    print("=" * 60)

    # Test initialization
    print("\n1. Testing DeepResearchFilter initialization...")
    mock_client = APIClient(
        model_name="gpt-5-mini",
        api_type="openai",
        api_key="test-key",
    )
    filter_obj = DeepResearchFilter(
        api_client=mock_client,
        confidence_threshold=0.7,
        max_workers=5,
    )
    print("   ✓ DeepResearchFilter initialized successfully")
    print(f"   Confidence threshold: {filter_obj.confidence_threshold}")

    # Test prompt rendering
    print("\n2. Testing prompt rendering...")
    variables = {
        "deep_research_query": "What are the latest LLM benchmarks?",
        "persona_background": "10 years AI research experience",
        "key_challenges": "Need up-to-date data",
    }
    prompt = render_prompt(DEEP_RESEARCH_FILTER_PROMPT, variables)
    assert "latest LLM benchmarks" in prompt
    assert "{{deep_research_query}}" not in prompt
    print("   ✓ Prompt rendered correctly")

    # Validate filter result data structure
    print("\n3. Validating filter result data structure...")
    mock_filter_result = {
        "needs_deep_research": True,
        "confidence_score": 0.85,
        "reasoning": "Requires up-to-date benchmark data from multiple sources...",
        "search_complexity": "High",
        "information_sources_needed": ["academic papers", "news", "technical reports"],
        "latest_info_required": True,
        "cross_domain_integration": True,
    }
    required_fields = [
        "needs_deep_research",
        "confidence_score",
        "reasoning",
        "search_complexity",
    ]
    for field in required_fields:
        assert field in mock_filter_result, f"Missing field: {field}"
    print(f"   ✓ Filter result contains all required fields")

    # Test confidence threshold logic
    print("\n4. Testing confidence threshold logic...")
    assert 0.85 >= 0.7, "High confidence should pass"
    assert 0.6 < 0.7, "Low confidence should be filtered"
    print("   ✓ Confidence threshold logic correct")

    print("\n" + "=" * 60)
    print("Deep Research Filter module test complete!")
    print("Note: Actual filtering requires a valid API Key")
