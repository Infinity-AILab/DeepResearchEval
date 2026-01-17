<div align="center">
  <h1 align="center">
    DeepResearch Eval: An Automated Framework for Deep Research Task Construction and Agentic Evaluation
  </h1>
  <p>
    &nbsp;&nbsp;🌐 <a href="https://infinity-ailab.github.io/deep_research_eval/">Website</a>&nbsp;&nbsp;|&nbsp;&nbsp;
    📑 <a href="https://arxiv.org/abs/2601.09688">Paper</a>&nbsp;&nbsp;|&nbsp;&nbsp;
    🤗 <a href="https://huggingface.co/datasets/Infinity-AILab/DeepResearchEval">Dataset</a>&nbsp;&nbsp;|&nbsp;&nbsp;
    🐥 <a href="https://docs.google.com/document/d/1tWxvyePIYO-yGIrcPUdyHgCieWv322EGtVyLROZgY2M/edit?tab=t.0">Submission</a>&nbsp;&nbsp;
  </p>
    <h5 align="center"> If you find this project useful, please give us a star🌟.

</div>


## 📰 News 
- **2025-01-17**: 🔥 We release DeepResearchEval [website](https://infinity-ailab.github.io/deep_research_eval/).
- **2025-01-15**: 🔥 We release the [DeepResearchEval]() and the [paper](https://arxiv.org/abs/2601.09688).


## 👋 Overview
-  We introduce **DeepResearchEval**, an automated framework for deep research task construction and agentic evaluation.
- For task construction, we propose a **persona-driven pipeline generating realistic, complex research tasks** anchored in diverse user profiles, applying a two-stage filter Task Qualification and Search Necessity to retain only tasks requiring multi-source evidence integration and external retrieval.
- For evaluation, we propose an agentic pipeline with two components: an **Adaptive Point-wise Quality Evaluation** that dynamically derives task-specific evaluation dimensions, criteria, and weights conditioned on each generated task, and an **Active Fact-Checking** that autonomously extracts and verifies report statements via web search, even when citations are missing. 

## 📊 Model Performance Ranking

<p align="center">
  <img src="model_ranking.png" width="800">
</p>

<p align="center">
  <em>
  Overview of deep research systems' performance on our benchmark. The left panel reports quality evaluation results across deep research systems, with Gemini-2.5-Pro achieving the highest score (8.51/10). The right panel reports factual correctness, where Manus achieves the highest ratio of correct statements (82.3%).
  </em>
</p>

## 📝 Task Generation

For installation,

We recommend using [`uv`](https://docs.astral.sh/uv/) with `python >= 3.10`

```bash
# Clone the repo
git clone https://github.com/Infinity-AILab/DeepResearchEval.git
cd DeepResearchEval

# Install dependencies and create virtual environment
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

After activation, you can run Python commands directly without `uv run` prefix.

Generate expert-level tasks that require deep web search and information synthesis.

```bash
# Run complete pipeline
python task_generation/main.py --output_file ./task_generation/outputs/deep_research_tasks.jsonl --model_name gpt-5-mini
```

For detailed usage, parameters, and examples, see [task_generation/README.md](task_generation/README.md).


## 💻 Evaluation

### Adaptive Point-wise Quality Evaluation

For installation,

```bash
cd poin_quality

pip install -r requirements.txt
```

For usage,
```bash
# To use google/gemini-2.5-pro-preview as the judge LLM
export OPENROUTER_API_KEY="your_openrouter_api_key"

cd poin_quality

python example_pointwise_usage.py
```
When running the script, the judging process follows this logic:

- If `criteria_cache.json`, `dimensions_cache.json`, and `weights_cache.json` already exist in `./point_quality/outputs/cache/`, the script will directly reuse the cached criteria, dimensions, and weights to perform point-wise judging.

- Otherwise, the script will first generate task-specific dimensions, criteria, and weights, cache them under `./point_quality/outputs/cache/`, and then proceed with the judging process.

The point-wise evaluation is configured via a YAML file located at:
```text
./point_quality/deepresearcharena/config/pointwise.yaml
```

You can modify the judge LLM settings under the `evaluator_model` field in the configuration file, including the model name and related parameters (e.g., temperature, max tokens).

The models (or methods) to be evaluated are specified under the target_models field.
For example, if your evaluation results are stored in: ./data/method_results/aaa/, ./data/method_results/bbb/ .
you should configure:
```text
target_models:
  - "aaa"
  - "bbb"
```

### Active Fact-Checking

For active fact-checking, we implement a fact-checking agent based on [MiroFlow](https://github.com/MiroMindAI/MiroFlow).

We recommend using [`uv`](https://docs.astral.sh/uv/) with `python >= 3.10`

Step1: prepare python environment:
```bash
# Run complete pipeline
cd factual_eval/apps/run-agent

uv sync
```
Step2: Set up environment dependencies:
```bash
cd factual_eval/apps/run-agent

vim .env
# Set the API KEY
# OPENROUTER_API_KEY (Using OpenRouter to provide primary agent model)
# OPENAI_API_KEY for openai models
# SERPER_API_KEY (for Google search and website scraping)
```
Step3: Fact-checking evaluation
```bash
cd factual_eval/apps/run-agent

uv run batch_test.py --json_dir ../../../data/method_results/gemini_2.5_pro # replace with your file name

# or runs the evaluation in the background and records logs to a log file:
bash batch_fact.sh
```
The configurations for the **framework**, **agent**, and **LLM** （default: gpt-5-mini） are defined under:

```text
./factual_eval/libs/miroflow/src/miroflow/prebuilt/config
```
You can check more details of our active fact-checking in factual_eval/README.md


## 🙏 Acknowledgement

We thank the [MiroFlow](https://github.com/MiroMindAI/MiroFlow) and [DAComp](https://github.com/ByteDance-Seed/DAComp) for their open source contribution. 

## ✍️ Citation
If you find our work helpful, please cite as
```
@misc{wang2026deepresearchevalautomatedframeworkdeep,
      title={DeepResearchEval: An Automated Framework for Deep Research Task Construction and Agentic Evaluation}, 
      author={Yibo Wang and Lei Wang and Yue Deng and Keming Wu and Yao Xiao and Huanjin Yao and Liwei Kang and Hai Ye and Yongcheng Jing and Lidong Bing},
      year={2026},
      eprint={2601.09688},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.09688}, 
}
```
