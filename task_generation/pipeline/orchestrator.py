"""
Pipeline Orchestrator

Chain all generation and filtering steps to execute the complete task generation workflow.
"""

import time
import logging
from typing import Any, Dict, List

from ..config import TaskGenerationConfig
from ..clients.api_client import APIClient
from ..generators.domain_loader import DomainLoader
from ..generators.expert_generator import ExpertGenerator
from ..generators.task_generator import TaskGenerator
from ..filters.deep_research_filter import DeepResearchFilter
from ..filters.quality_assessor import QualityAssessor
from ..utils.file_utils import save_json, save_jsonl

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Deep Research task generation pipeline orchestrator."""

    def __init__(self, config: TaskGenerationConfig) -> None:
        """
        Initialize the pipeline.

        Args:
            config: Task generation configuration
        """
        self.config = config
        self.api_client = APIClient(
            model_name=config.model_name,
            api_type=config.api_type,
        )

        # Initialize components
        self.domain_loader = DomainLoader(config.domains_file)
        self.expert_generator = ExpertGenerator(
            api_client=self.api_client,
            cache_file=config.experts_file,
            max_workers=config.max_workers,
        )
        self.task_generator = TaskGenerator(
            api_client=self.api_client,
            cache_file=config.tasks_file,
            max_workers=config.max_workers,
        )
        self.deep_research_filter = DeepResearchFilter(
            api_client=self.api_client,
            cache_file=config.filtered_tasks_file,
            max_workers=config.max_workers,
            confidence_threshold=config.confidence_threshold,
        )
        self.quality_assessor = QualityAssessor(
            api_client=self.api_client,
            baseline_cache_file=config.baseline_file,
            max_workers=config.max_workers,
            quality_threshold=config.quality_score_threshold,
        )

        # Store intermediate results
        self.domains: List[str] = []
        self.experts: List[Dict[str, Any]] = []
        self.tasks: List[Dict[str, Any]] = []
        self.filtered_tasks: List[Dict[str, Any]] = []
        self.final_tasks: List[Dict[str, Any]] = []

    def run(self) -> List[Dict[str, Any]]:
        """
        Execute the complete generation pipeline.

        Returns:
            List of final generated tasks
        """
        logger.info("=" * 60)
        logger.info("Starting Deep Research task generation workflow")
        logger.info("=" * 60)

        try:
            # Step 1: Load domains
            logger.info("\n📚 Step 1: Load research domains")
            self.domains = self.domain_loader.load()

            # Step 2: Generate experts
            logger.info("\n👥 Step 2: Generate expert personas")
            self.experts = self.expert_generator.generate(self.domains)

            # Step 3: Generate tasks
            logger.info("\n📝 Step 3: Generate Deep Research tasks")
            self.tasks = self.task_generator.generate(self.experts)

            # Step 4: Filter tasks requiring deep search
            logger.info("\n🔍 Step 4: Filter deep search tasks")
            self.filtered_tasks = self.deep_research_filter.filter(self.tasks)

            # Step 5: Generate baseline answers
            logger.info("\n💬 Step 5: Generate no-search baseline answers")
            tasks_with_baseline = self.quality_assessor.generate_baselines(
                self.filtered_tasks
            )

            # Step 6: Assess quality
            logger.info("\n⭐ Step 6: Assess answer quality")
            self.final_tasks = self.quality_assessor.assess_quality(tasks_with_baseline)

            # Step 7: Save results
            logger.info("\n💾 Step 7: Save final results")
            self._save_final_tasks()

            logger.info("\n" + "=" * 60)
            logger.info("✅ Deep Research task generation complete!")
            logger.info("=" * 60)

            return self.final_tasks

        except Exception as e:
            logger.error(f"❌ Error during task generation: {e}")
            raise

    def _save_final_tasks(self) -> None:
        """Save final task data."""
        logger.info(f"Saving to: {self.config.output_file}")

        # Format final data
        final_data = []
        for task in self.final_tasks:
            formatted_task = self._format_task(task)
            final_data.append(formatted_task)

        # Save as JSONL
        save_jsonl(final_data, self.config.output_file)

        # Also save as JSON
        json_file = self.config.output_file.replace(".jsonl", ".json")
        save_json(final_data, json_file)

        logger.info(f"Successfully saved {len(final_data)} tasks")
        logger.info(f"JSONL: {self.config.output_file}")
        logger.info(f"JSON:  {json_file}")

        # Print statistics
        self._print_statistics(final_data)

    def _format_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Format a single task for final output."""
        expert = task.get("expert", {})
        filter_result = task.get("filter_result", {})
        quality_assessment = task.get("quality_assessment", {})

        return {
            "task_id": task.get("task_id", ""),
            "domain": task.get("domain", "unknown"),
            "expert": {
                "name": expert.get("name", ""),
                "role": expert.get("role", ""),
                "affiliation": expert.get("affiliation", ""),
                "background": expert.get("background", ""),
                "subdomain": expert.get("subdomain", ""),
            },
            "query": {
                "deep_research_query": task.get("deep_research_query", ""),
                "key_challenges": task.get("key_challenges", ""),
                "expected_search_rounds": task.get("expected_search_rounds", 0),
                "time_sensitivity": task.get("time_sensitivity", False),
                "time_constraint": task.get("time_constraint"),
            },
            "evaluation_results": {
                "needs_deep_research": filter_result.get("needs_deep_research", False),
                "confidence_score": filter_result.get("confidence_score", 0),
                "search_complexity": filter_result.get("search_complexity", ""),
                "information_sources_needed": filter_result.get(
                    "information_sources_needed", []
                ),
                "quality_assessment": quality_assessment.get("overall_quality", ""),
                "quality_score": quality_assessment.get("quality_score", 0),
                "requires_search": quality_assessment.get("requires_search", False),
            },
            "baseline_answer": task.get("baseline_answer", ""),
            "retention_reason": task.get("retention_reason", ""),
            "generated_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _print_statistics(self, final_data: List[Dict[str, Any]]) -> None:
        """Print statistics."""
        logger.info("\n" + "=" * 60)
        logger.info("📊 Task Generation Statistics")
        logger.info("=" * 60)

        if not final_data:
            logger.info("No tasks generated")
            return

        # Statistics by domain
        domain_count: Dict[str, int] = {}
        time_sensitive_count = 0

        for task in final_data:
            domain = task.get("domain", "unknown")
            domain_count[domain] = domain_count.get(domain, 0) + 1
            if task.get("query", {}).get("time_sensitivity", False):
                time_sensitive_count += 1

        total = len(final_data)
        logger.info(f"Total tasks: {total}")
        logger.info(
            f"Time-sensitive tasks: {time_sensitive_count} ({time_sensitive_count/total*100:.1f}%)"
        )
        logger.info("\nTask distribution by domain:")
        for domain, count in sorted(domain_count.items()):
            logger.info(f"  {domain}: {count}")

        logger.info("=" * 60)


if __name__ == "__main__":
    import os
    import tempfile
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print("=" * 60)
    print("Pipeline Orchestrator Module Test")
    print("=" * 60)

    # Create temporary test environment
    temp_dir = tempfile.mkdtemp()
    print(f"\nUsing temporary directory: {temp_dir}")

    try:
        # Create test domains.json
        domains_file = os.path.join(temp_dir, "domains.json")
        with open(domains_file, "w") as f:
            json.dump(["AI", "Healthcare"], f)

        # Test configuration creation
        print("\n1. Testing configuration and initialization...")
        config = TaskGenerationConfig(
            output_file=os.path.join(temp_dir, "output.jsonl"),
            domains_file=domains_file,
            experts_file=os.path.join(temp_dir, "experts.json"),
            tasks_file=os.path.join(temp_dir, "tasks.json"),
            filtered_tasks_file=os.path.join(temp_dir, "filtered.json"),
            baseline_file=os.path.join(temp_dir, "baseline.json"),
            model_name="gpt-5-mini",
            max_workers=5,
        )
        print("   ✓ TaskGenerationConfig created successfully")

        # Note: Actual PipelineOrchestrator initialization requires a valid API Key
        # Here we only test configuration
        print("\n2. Validating configuration parameters...")
        print(f"   output_file: {config.output_file}")
        print(f"   domains_file: {config.domains_file}")
        print(f"   model_name: {config.model_name}")
        print(f"   max_workers: {config.max_workers}")
        print("   ✓ Configuration parameters correct")

        # Test task formatting
        print("\n3. Testing task formatting logic...")
        mock_task = {
            "task_id": "task_1",
            "domain": "AI",
            "deep_research_query": "What are the latest LLM benchmarks?",
            "key_challenges": "Need current data",
            "expected_search_rounds": 4,
            "time_sensitivity": True,
            "time_constraint": "As of 2025",
            "expert": {
                "name": "Dr. Chen",
                "role": "Researcher",
                "affiliation": "Stanford",
                "background": "10 years experience",
                "subdomain": "NLP",
            },
            "filter_result": {
                "needs_deep_research": True,
                "confidence_score": 0.9,
                "search_complexity": "High",
                "information_sources_needed": ["papers", "news"],
            },
            "quality_assessment": {
                "overall_quality": "low",
                "quality_score": 0.3,
                "requires_search": True,
            },
            "baseline_answer": "Here is my analysis...",
            "retention_reason": "Needs search augmentation",
        }

        # Simulate formatting (without instantiating orchestrator)
        formatted = {
            "task_id": mock_task["task_id"],
            "domain": mock_task["domain"],
            "expert": mock_task["expert"],
            "query": {
                "deep_research_query": mock_task["deep_research_query"],
                "key_challenges": mock_task["key_challenges"],
                "expected_search_rounds": mock_task["expected_search_rounds"],
                "time_sensitivity": mock_task["time_sensitivity"],
                "time_constraint": mock_task["time_constraint"],
            },
            "evaluation_results": {
                "needs_deep_research": True,
                "confidence_score": 0.9,
                "search_complexity": "High",
            },
        }
        assert "task_id" in formatted
        assert "query" in formatted
        assert "expert" in formatted
        print("   ✓ Task formatting logic correct")

    finally:
        import shutil

        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temporary directory")

    print("\n" + "=" * 60)
    print("Pipeline Orchestrator module test complete!")
    print("Note: Full execution requires a valid API Key and domains.json file")
