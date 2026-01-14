"""
Prompt templates for Pointwise Evaluation
"""

# Dimension generation prompt
DIMENSION_GENERATION_PROMPT = """
<system_role>
You are an expert evaluator who designs **query-specific meta-evaluation dimensions** for deep research reports. Your goal is to identify unique quality aspects that matter for a given task, beyond the four standard meta-dimensions.
</system_role>

<user_prompt>
**Standard Meta-Dimensions** (already covered):
1. **Coverage**: Breadth, depth, and relevance of coverage
2. **Insight**: Depth, originality, logic, and value of analysis
3. **Instruction Following**: Accuracy in meeting all requirements
4. **Clarity**: Clarity, fluency, structure, and ease of understanding

**Your Task**: For the research task below, generate **1–3 additional same-level meta-evaluation dimensions** that are:
- Highly specific to this query
- Distinct from the four standard meta-dimensions
- Crucial for assessing quality in this domain
- Actionable and measurable
- Serve as **upper-level meta-dimensions** that can be further expanded into more detailed evaluation standards 
- Do NOT include any factuality-related meta-dimensions, since factual accuracy is handled by a separate evaluation system

<research_task>
"{task_prompt}"
</research_task>

**Guidelines**:
1. Analyze the task carefully to understand its domain, methodology, data needs, and unique challenges.
2. Identify domain-specific quality factors (e.g., for finance: market timing; for science: experimental validity; for policy: stakeholder impact).
3. Create meta-dimensions that are:
   - Unique to this query type (not generic)
   - Non-overlapping with the four standard meta-dimensions
   - Focused on specialized aspects relevant to this domain
4. For each meta-dimension, provide:
   - **Name** (1–3 words)
   - **Definition** (short explanation of what it measures and why it matters)

**Output Format**:
Return only a JSON list of meta-dimensions, e.g.:

<json_output>
[
  {{
    "meta_dimension_name": "Xxx",
    "definition": "Clear, concise explanation"
  }},
  ...
]
</json_output>

</user_prompt>
"""

# Dimension weights generation prompt
WEIGHT_GENERATION_PROMPT = """
<system_role>
You are a senior research evaluation expert. Your job is to (1) consider both the four fixed meta-dimensions and the provided query-specific meta-dimensions (each with a name and definition), and (2) assign **dynamic, well-justified weights** to all dimensions so that the total equals **1.0**.
</system_role>

<user_prompt>
There is a deep research task as follows:
<task>
"{task_prompt}"
</task>

**Fixed Meta-Dimensions (always included):**
[
  {{
    "meta_dimension_name": "Coverage",
    "definition": "Breadth, depth, and relevance of coverage."
  }},
  {{
    "meta_dimension_name": "Insight",
    "definition": "Depth, originality, logic, and value of analysis."
  }},
  {{    
    "meta_dimension_name": "Instruction Following",
    "definition": "Accuracy in meeting all requirements and constraints."
  }},
  {{
    "meta_dimension_name": "Clarity",
    "definition": "Readability, fluency, structure, and ease of understanding."
  }}
]

**Provided Query-Specific Meta-Dimensions (each includes name + definition)**
<additional_meta_dimensions_json>
{additional_dimensions_json}
</additional_meta_dimensions_json>

**Your Goals**
1. **Task-grounded analysis**: Carefully analyze the <task> to identify goals, constraints, risks, and success criteria.
2. **Dynamic weighting**: Assign a weight (0–1) to **each** dimension (fixed + provided).  
   - The **sum across all dimensions must be exactly 1.0**.
   - Weights should reflect the **unique characteristics** of the <task> (do not use fixed presets).
3. **Specific justification**: In <analysis>, explicitly justify **each** weight by referencing the <task> and, for the provided meta-dimensions, their **definitions**. Avoid generic statements.

**Constraints & Notes**
- Do **not** introduce new dimensions. Only use the fixed four plus the provided query-specific ones.
- Do **not** include any factuality-related dimensions (factuality is evaluated elsewhere).
- Avoid overlap: if a provided dimension substantially overlaps with a fixed one, explain how you differentiate it in scope and why its weight is still necessary.
- If no additional dimensions are provided (empty list), distribute weights among the four fixed dimensions only.

**Output Format (STRICT)**
First produce a concise yet concrete <analysis> that:
- Explains the task-grounded reasoning behind the overall weighting strategy.
- Gives a 1–3 sentence justification **for each dimension** tying the weight to the task and (for provided ones) their definitions.

Then produce <json_output> containing only the final weights, with keys exactly matching the dimension names:
- Fixed keys (always present): "comprehensiveness", "insight", "instruction_following", "readability"
- For each provided meta-dimension, use its exact "meta_dimension_name" as the key.

Example shape:
<analysis>
(Your reasoning here. One short paragraph about overall trade-offs, then bullet points or short lines justifying each dimension's weight.)
</analysis>

<json_output>
{{
  "coverage": 0.xx,
  "insight": 0.xx,
  "instruction_following": 0.xx,
  "clarity": 0.xx,
  "additional_dimension": 0.xx
}}
</json_output>

**Validation**
- Ensure all weights are in [0, 1].
- Ensure the sum across all keys equals **1.00** (allowing up to ±0.001 rounding; otherwise adjust and state the adjustment briefly in <analysis>).
- Do not output anything other than <analysis> and <json_output>.
</user_prompt>
"""

# Criteria generation prompt
CRITERIA_GENERATION_PROMPT = """
<system_role>
You are an expert evaluator of research reports. Your job is to break down an meta-evaluation dimension into clear, specific, task-relevant criteria with explanations and weights.
</system_role>

<user_prompt>
We evaluate a research report written for the task below across {num_dimensions} meta evaluation dimensions:
{meta_dimensions}

<task>
"{task_prompt}"
</task>

<instruction>
Your goal: For the **{dimension_name}** dimension, generate task-specific evaluation criteria.

Steps:
1. **Analyze Task**: Identify the essential areas and coverage needed to satisfy "{dimension_name}".
2. **Formulate Criteria**: Write diverse, non-overlapping criteria items.
3. **Explain Rationale**: Provide a short explanation (`explanation`) for each criterion.
4. **Assign Weights**: Give each criterion a weight (`weight`) so that the total = **1.0**. Adjust the last item if needed.
5. **Focus**: Stay strictly within "{dimension_name}", avoiding overlap with the other dimensions.

Output format:
1. First, provide `<analysis>` explaining your reasoning and weight allocation.
2. Then output `<json_output>` as a list of criteria in the format:

<json_output>
[
  {{
    "criterion": "...",
    "explanation": "...",
    "weight": ...
  }},
  ...
]
</json_output>

Now begin for the task above and dimension = **{dimension_name}**.
</user_prompt>
"""


SCORING_PROMPT = """
<system_role>
You are a strict, meticulous, and objective evaluator of deep research reports. 
You score the report on a **single evaluation dimension** at a time.  
You must evaluate strictly according to the provided **criteria under that dimension**. 
Do **not** evaluate factual accuracy (handled by a separate system).
</system_role>

<user_prompt>
**Task**
<task>
{task_prompt}
</task>

**Report to Evaluate**
<Report>
{report}
</Report>

**Evaluation Dimension and Criteria**
Below is the dimension to evaluate. It contains multiple criteria, each with a short explanation.  
<criteria_of_one_dimension_json>
{criteria_of_one_dimension_json}
</criteria_of_one_dimension_json>

**Scoring Rules**
- For **each criterion**, assign a **continuous score from 0 to 10** (real number), and provide a concise justification (`analysis`) grounded in the report content.
- **Revised Scale (more discriminative):**
  - **0–2 (Very Poor):** Severe deficiencies; almost entirely fails the criterion.
  - **2–4 (Poor):** Major gaps or flaws; only marginally addresses the criterion.
  - **4–6 (Fair):** Covers the basics but shallow, inconsistent, or with notable weaknesses.
  - **6–7.5 (Good):** Solid attempt; generally clear and adequate, but lacks depth or polish.
  - **7.5–9 (Very Good):** Strong and well-executed with only minor issues; above average.
  - **9–10 (Excellent):** Outstanding; fully satisfies and exceeds expectations in depth, clarity, and execution. Reserved for exceptional cases.
- Scores should reflect only the current criterion, avoiding overlap with other dimensions.
- **Do not** judge factual correctness; only judge coverage, reasoning quality, clarity, structure, domain-appropriateness, etc.
- Be conservative: most typical reports should fall in the **4–8 range**. Scores above 9 should be rare.

**Output Format (STRICT)**
Output `<json_output>` as a list of criteria in a valid JSON object format:


<json_output>
[
    {{
        "criterion": "text of the criterion",
        "analysis": "your justification",
        "report_score_0_to_10": x.xx
    }},
    {{
        "criterion": "text of the criterion",
        "analysis": "your justification",
        "report_score_0_to_10": x.xx
    }},
    ...
]
</json_output>

**Validation**
- Use the exact criterion names as keys in the JSON.
- Each score must be a real number in [0,10], rounded to two decimals.
- Ensure the JSON is strictly valid and parseable (no trailing commas, properly escaped characters).
</user_prompt>
"""
