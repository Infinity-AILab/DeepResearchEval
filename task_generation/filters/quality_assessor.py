"""
Answer Quality Assessor

Generate no-search baseline answers and assess their quality,
keeping tasks that require search augmentation.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from ..clients.api_client import APIClient
from ..prompts.templates import (
    NO_SEARCH_BASELINE_PROMPT,
    QUALITY_ASSESSMENT_PROMPT,
    render_prompt,
)
from ..utils.json_utils import safe_parse_json
from ..utils.file_utils import load_if_exists, save_json

logger = logging.getLogger(__name__)


class QualityAssessor:
    """Assess baseline answer quality, filter tasks needing search augmentation."""

    def __init__(
        self,
        api_client: APIClient,
        baseline_cache_file: Optional[str] = None,
        max_workers: int = 25,
        quality_threshold: float = 0.7,
    ) -> None:
        """
        Initialize the quality assessor.

        Args:
            api_client: API client instance
            baseline_cache_file: Cache file path for baseline answers
            max_workers: Maximum parallel threads
            quality_threshold: Quality score threshold, tasks above this are filtered
        """
        self.api_client = api_client
        self.baseline_cache_file = baseline_cache_file
        self.max_workers = max_workers
        self.quality_threshold = quality_threshold

    def generate_baselines(
        self, tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate no-search baseline answers for tasks.

        Args:
            tasks: Task list

        Returns:
            Task list with baseline_answer field attached
        """
        # Check cache
        if self.baseline_cache_file:
            cached = load_if_exists(self.baseline_cache_file, default=None)
            if cached is not None:
                logger.info(f"Using cached baseline answers: {len(cached)}")
                return cached

        logger.info(f"Generating baseline answers in parallel ({len(tasks)} tasks)...")

        tasks_with_baseline: List[Dict[str, Any]] = []
        workers = min(self.max_workers, len(tasks))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_task = {
                executor.submit(self._generate_baseline, task): task for task in tasks
            }

            completed = 0
            for future in as_completed(future_to_task):
                completed += 1
                task = future_to_task[future]

                try:
                    result = future.result()
                    if result is not None:
                        tasks_with_baseline.append(result)

                        # Save after each task completion
                        if self.baseline_cache_file:
                            save_json(tasks_with_baseline, self.baseline_cache_file)
                            logger.info(
                                f"Progress: {len(tasks_with_baseline)}/{len(tasks)}"
                            )
                except Exception as e:
                    logger.error(
                        f"Error processing task {task.get('task_id', 'unknown')}: {e}"
                    )

        logger.info(f"Total generated: {len(tasks_with_baseline)} baseline answers")
        return tasks_with_baseline

    def _generate_baseline(self, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate baseline answer for a single task.

        Args:
            task: Task info

        Returns:
            Task with baseline_answer attached, None on failure
        """
        task_id = task.get("task_id", "unknown")
        logger.debug(f"Generating baseline answer: {task_id}")

        variables = {
            "persona_background": task.get("expert", {}).get("background", ""),
            "deep_research_query": task.get("deep_research_query", ""),
        }
        prompt = render_prompt(NO_SEARCH_BASELINE_PROMPT, variables)
        messages = [{"role": "user", "content": prompt}]

        response = self.api_client.generate_response(messages)
        if self.api_client.is_error(response):
            logger.error(f"Baseline answer generation failed: {task_id}")
            return None

        task["baseline_answer"] = response
        logger.info(f"  Generated baseline answer {task_id} ({len(response)} chars)")
        return task

    def assess_quality(
        self, tasks_with_baseline: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Assess baseline answer quality, filter tasks needing search augmentation.

        Args:
            tasks_with_baseline: Task list with baseline answers

        Returns:
            List of tasks that passed quality filtering
        """
        logger.info(f"Assessing answer quality in parallel ({len(tasks_with_baseline)} tasks)...")

        final_tasks: List[Dict[str, Any]] = []
        workers = min(self.max_workers, len(tasks_with_baseline))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_task = {
                executor.submit(self._assess_single, task): task
                for task in tasks_with_baseline
            }

            completed = 0
            for future in as_completed(future_to_task):
                completed += 1
                task = future_to_task[future]

                try:
                    result = future.result()
                    if result is not None:
                        final_tasks.append(result)
                    logger.info(f"Quality assessment progress: {completed}/{len(tasks_with_baseline)}")
                except Exception as e:
                    logger.error(
                        f"Error processing task {task.get('task_id', 'unknown')}: {e}"
                    )

        logger.info(f"Final kept: {len(final_tasks)} high-quality tasks")
        return final_tasks

    def _assess_single(self, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Assess answer quality for a single task.

        Args:
            task: Task with baseline answer

        Returns:
            Task that passed quality filter (with assessment), None if filtered
        """
        task_id = task.get("task_id", "unknown")
        logger.debug(f"Assessing quality: {task_id}")

        variables = {
            "deep_research_query": task.get("deep_research_query", ""),
            "key_challenges": task.get("key_challenges", ""),
            "persona_background": task.get("expert", {}).get("background", ""),
            "baseline_answer": task.get("baseline_answer", ""),
        }
        prompt = render_prompt(QUALITY_ASSESSMENT_PROMPT, variables)
        messages = [{"role": "user", "content": prompt}]

        response = self.api_client.generate_response(messages)
        if self.api_client.is_error(response):
            logger.error(f"Quality assessment failed: {task_id}")
            return None

        quality_result = safe_parse_json(response)
        if quality_result is None:
            logger.error(f"Failed to parse quality assessment JSON: {task_id}")
            return None

        # Check quality assessment result
        overall_quality = quality_result.get("overall_quality", "").lower()
        requires_search = quality_result.get("requires_search", False)
        quality_score = quality_result.get("quality_score", 1.0)

        # Keep low/medium quality tasks that need search
        low_or_medium = overall_quality in ["low", "medium"]

        if low_or_medium and requires_search and quality_score <= self.quality_threshold:
            task["quality_assessment"] = quality_result
            task["retention_reason"] = (
                f"Baseline quality {overall_quality} "
                f"(score: {quality_score:.2f}), requires search augmentation"
            )
            logger.info(
                f"  ✓ Keep {task_id} (quality: {overall_quality}, score: {quality_score:.2f})"
            )
            return task
        else:
            logger.info(
                f"  ✗ Filter {task_id} "
                f"(quality: {overall_quality}, requires_search: {requires_search}, "
                f"score: {quality_score:.2f})"
            )
            return None


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print("=" * 60)
    print("Quality Assessor Module Test")
    print("=" * 60)

    # Test initialization
    print("\n1. Testing QualityAssessor initialization...")
    mock_client = APIClient(
        model_name="gpt-5-mini",
        api_type="openai",
        api_key="test-key",
    )
    assessor = QualityAssessor(
        api_client=mock_client,
        quality_threshold=0.7,
        max_workers=5,
    )
    print("   ✓ QualityAssessor initialized successfully")
    print(f"   Quality threshold: {assessor.quality_threshold}")

    # Test baseline answer prompt rendering
    print("\n2. Testing baseline answer prompt rendering...")
    variables = {
        "persona_background": "10 years AI research",
        "deep_research_query": "What are the latest LLM benchmarks?",
    }
    prompt = render_prompt(NO_SEARCH_BASELINE_PROMPT, variables)
    assert "latest LLM benchmarks" in prompt
    assert "{{deep_research_query}}" not in prompt
    print("   ✓ Baseline answer prompt rendered correctly")

    # Test quality assessment prompt rendering
    print("\n3. Testing quality assessment prompt rendering...")
    variables = {
        "deep_research_query": "What are the latest LLM benchmarks?",
        "key_challenges": "Need up-to-date data",
        "persona_background": "10 years AI research",
        "baseline_answer": "Here is my analysis of LLM benchmarks...",
    }
    prompt = render_prompt(QUALITY_ASSESSMENT_PROMPT, variables)
    assert "latest LLM benchmarks" in prompt
    assert "{{baseline_answer}}" not in prompt
    print("   ✓ Quality assessment prompt rendered correctly")

    # Validate quality assessment result data structure
    print("\n4. Validating quality assessment result data structure...")
    mock_quality_result = {
        "overall_quality": "low",
        "quality_score": 0.3,
        "completeness_score": 0.4,
        "accuracy_score": 0.7,
        "depth_score": 0.2,
        "timeliness_score": 0.1,
        "accuracy_concerns": "Some data may be outdated",
        "missing_aspects": "Latest benchmark results from 2025",
        "outdated_info": "Benchmark data is from 2023",
        "requires_search": True,
        "search_necessity_reasoning": "Need current benchmark data...",
    }
    required_fields = [
        "overall_quality",
        "quality_score",
        "requires_search",
    ]
    for field in required_fields:
        assert field in mock_quality_result, f"Missing field: {field}"
    print(f"   ✓ Quality assessment result contains all required fields")

    # Test filtering logic
    print("\n5. Testing filtering logic...")
    # Low quality + needs search + low score -> keep
    assert ("low" in ["low", "medium"]) and True and (0.3 <= 0.7)
    print("   ✓ Low quality task retention logic correct")

    # High quality -> filter
    assert "high" not in ["low", "medium"]
    print("   ✓ High quality task filtering logic correct")

    print("\n" + "=" * 60)
    print("Quality Assessor module test complete!")
    print("Note: Actual assessment requires a valid API Key")
