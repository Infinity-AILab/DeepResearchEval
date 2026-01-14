"""Task generation configuration management."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class TaskGenerationConfig:
    """Configuration parameters for Deep Research task generation."""

    # Output paths
    output_file: str = "./outputs/task_generation/deep_research_tasks.jsonl"

    # Generation parameters
    num_domains: int = 10
    tasks_per_expert: int = 3

    # LLM configuration
    model_name: str = "gpt-5-mini"
    api_type: Literal["openai", "openrouter"] = "openai"
    max_retries: int = 3
    max_tokens: int = 8192

    # Parallel configuration
    max_workers: int = 25

    # Input file paths
    domains_file: str = "./task_generation/domains.json"
    
    # Intermediate file paths
    experts_file: str = "./outputs/task_generation/intermediate_2_experts.json"
    tasks_file: str = "./outputs/task_generation/intermediate_3_tasks.json"
    filtered_tasks_file: str = "./outputs/task_generation/intermediate_4_filtered_tasks.json"
    baseline_file: str = "./outputs/task_generation/intermediate_5_baseline_answers.json"

    # Filtering thresholds
    confidence_threshold: float = 0.7
    quality_score_threshold: float = 0.7


if __name__ == "__main__":
    # Test configuration creation
    config = TaskGenerationConfig(
        output_file="./test_output.jsonl",
        model_name="gpt-4",
        num_domains=5,
    )
    print("✓ TaskGenerationConfig created successfully")
    print(f"  output_file: {config.output_file}")
    print(f"  model_name: {config.model_name}")
    print(f"  num_domains: {config.num_domains}")
    print(f"  max_workers: {config.max_workers}")
    print(f"  confidence_threshold: {config.confidence_threshold}")
