# 📝 Deep Research Task Generator

Generate expert-level tasks that require deep web search and information synthesis.

## Directory Structure

```
task_generation/
├── main.py                   # CLI entry point
├── config.py                 # Configuration management
├── prompts/
│   └── templates.py          # All LLM prompt templates
├── clients/
│   └── api_client.py         # OpenAI/OpenRouter API client
├── generators/
│   ├── domain_loader.py      # Domain loader
│   ├── expert_generator.py   # Expert persona generator
│   └── task_generator.py     # Deep Research task generator
├── filters/
│   ├── deep_research_filter.py  # Deep research requirement filter
│   └── quality_assessor.py      # Answer quality assessor
├── pipeline/
│   └── orchestrator.py       # Complete pipeline orchestration
└── utils/
    ├── json_utils.py         # JSON parsing utilities
    └── file_utils.py         # File IO utilities
```

## Quick Start

```bash
# Set API Key
export OPENROUTER_KEY="your-api-key"

# Run task generation
python task_generation/main.py --output_file ./task_generation/outputs/deep_research_tasks.jsonl --model_name gpt-5-mini
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--output_file` | `./outputs/task_generation/deep_research_tasks.jsonl` | Output file path |
| `--num_domains` | 10 | Number of domains to use |
| `--tasks_per_expert` | 2 | Number of tasks per expert |
| `--model_name` | gpt-5-mini | Model name (e.g., `google/gemini-2.5-flash`) |
| `--api_type` | openai | API type: `openai` or `openrouter` |
| `--max_workers` | 25 | Number of parallel workers |
| `--confidence_threshold` | 0.7 | Deep search confidence threshold |
| `--quality_threshold` | 0.7 | Quality score threshold |
| `--verbose` | False | Enable verbose logging |

## Examples

```bash
# Small-scale test (2 domains)
python -m task_generation.main \
    --output_file ./task_generation/outputs/task_generation/test_tasks.jsonl \
    --num_domains 2 \
    --api_type openrouter \
    --model_name google/gemini-2.5-flash \
    --verbose

# Full-scale run (10 domains)
python -m task_generation.main \
    --output_file ./task_generation/outputs/task_generation/deep_research_tasks.jsonl \
    --num_domains 10 \
    --api_type openrouter \
    --model_name google/gemini-2.5-flash \
    --verbose

# Adjust thresholds if too many tasks are filtered
python -m task_generation.main \
    --api_type openrouter \
    --model_name google/gemini-2.5-flash \
    --confidence_threshold 0.5 \
    --quality_threshold 0.8 \
    --verbose
```

## Output

Final tasks are saved to the specified output file in JSONL format. Intermediate files are saved to `./outputs/task_generation/`:

- `intermediate_2_experts.json` - Expert profiles
- `intermediate_3_tasks.json` - Generated tasks
- `intermediate_4_filtered_tasks.json` - Filtered tasks
- `intermediate_5_baseline_answers.json` - Tasks with baseline answers

## Module Overview

| Module | Responsibility |
|--------|----------------|
| `config.py` | Task generation configuration dataclass |
| `prompts/templates.py` | Centralized LLM prompt template management |
| `clients/api_client.py` | LLM API calls with retry and error handling |
| `generators/domain_loader.py` | Load research domains from JSON file |
| `generators/expert_generator.py` | Parallel expert persona generation |
| `generators/task_generator.py` | Generate research tasks based on expert backgrounds |
| `filters/deep_research_filter.py` | Filter tasks truly requiring deep search |
| `filters/quality_assessor.py` | Assess baseline answer quality, keep tasks needing search |
| `pipeline/orchestrator.py` | Chain all steps into complete workflow |
| `utils/json_utils.py` | JSON extraction and parsing |
| `utils/file_utils.py` | File read/write with incremental saving |

## Individual Module Testing

Each module contains an `if __name__ == "__main__":` block for standalone testing:

```bash
python task_generation/config.py                      # Test configuration
python task_generation/prompts/templates.py           # Test prompt rendering
python task_generation/clients/api_client.py          # Test API client
python task_generation/generators/domain_loader.py    # Test domain loading
python task_generation/utils/json_utils.py            # Test JSON utilities
```
