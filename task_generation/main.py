#!/usr/bin/env python3
"""
Deep Research Task Generator - CLI Entry Point

Generate expert-level tasks that require deep web search and information synthesis.

Usage:
    python -m task_generation.main --output_file ./output.jsonl --api_type openrouter
    python -m task_generation.main --help
"""

import argparse
import logging
import sys
from typing import Literal

from .config import TaskGenerationConfig
from .pipeline.orchestrator import PipelineOrchestrator


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Deep Research Task Generator - Generate expert-level deep research tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --output_file ./tasks.jsonl
  python main.py --model_name gpt-4 --max_workers 10
  python main.py --verbose
        """,
    )

    # Output configuration
    parser.add_argument(
        "--output_file",
        type=str,
        default="./outputs/task_generation/deep_research_tasks.jsonl",
        help="Output file path (default: ./outputs/task_generation/deep_research_tasks.jsonl)",
    )

    # Generation parameters
    parser.add_argument(
        "--num_domains",
        type=int,
        default=10,
        help="Number of domains to generate (default: 10)",
    )
    parser.add_argument(
        "--tasks_per_expert",
        type=int,
        default=2,
        help="Number of tasks per expert (default: 2)",
    )

    # LLM configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-5-mini",
        help="LLM model to use (default: gpt-5-mini)",
    )
    parser.add_argument(
        "--api_type",
        type=str,
        choices=["openai", "openrouter"],
        default="openai",
        help="API type (default: openai)",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="API retry count (default: 3)",
    )

    # Parallel configuration
    parser.add_argument(
        "--max_workers",
        type=int,
        default=25,
        help="Maximum parallel workers (default: 25)",
    )

    # Filtering thresholds
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.7,
        help="Deep search confidence threshold (default: 0.7)",
    )
    parser.add_argument(
        "--quality_threshold",
        type=float,
        default=0.7,
        help="Quality score threshold (default: 0.7)",
    )

    # Other options
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging output",
    )
    parser.add_argument(
        "--domains_file",
        type=str,
        default="./task_generation/domains.json",
        help="Domains list file path (default: ./task_generation/domains.json)",
    )

    return parser.parse_args()


def main() -> int:
    """Main function."""
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    try:
        # Create configuration
        config = TaskGenerationConfig(
            output_file=args.output_file,
            num_domains=args.num_domains,
            tasks_per_expert=args.tasks_per_expert,
            model_name=args.model_name,
            api_type=args.api_type,
            max_retries=args.max_retries,
            max_workers=args.max_workers,
            confidence_threshold=args.confidence_threshold,
            quality_score_threshold=args.quality_threshold,
            domains_file=args.domains_file,
        )

        logger.info("Configuration:")
        logger.info(f"  Model: {config.model_name}")
        logger.info(f"  API Type: {config.api_type}")
        logger.info(f"  Output File: {config.output_file}")
        logger.info(f"  Domains File: {config.domains_file}")
        logger.info(f"  Max Workers: {config.max_workers}")

        # Create and run pipeline
        orchestrator = PipelineOrchestrator(config)
        final_tasks = orchestrator.run()

        logger.info(f"\n🎉 Successfully generated {len(final_tasks)} tasks!")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("\n⚠️ User interrupted")
        return 130
    except Exception as e:
        logger.exception(f"Error occurred: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
