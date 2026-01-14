"""
LLM Prompt Template Collection

All templates use {{variable}} as placeholders, rendered via render_prompt() function.
"""

from typing import Dict, Any


# =============================================================================
# Expert Generation Prompt
# =============================================================================

PERSONA_GENERATION_PROMPT = """You are a persona generator. For the domain "{{domain}}", generate 5 distinct personas.

Requirements:
1. Personas can be ordinary individuals (e.g., students, hobbyists, employees, community members) or domain experts.
2. Each persona must include a clear background: education, career path, interests, or life experience (100-150 words).
3. Each persona should have specific interests or responsibilities related to the domain.

Example subdomains:
Domain: Software Development
Subdomain: Artificial Intelligence/Machine Learning Development & Tools

Domain: Science & Technology
Subdomain: Large Language Models 
Subdomain: Artificial Intelligence & Machine Learning

Domain: Sports & Fitness
Subdomain: Professional Sports (Football, Basketball, etc.)

Strictly output the following JSON only, with no extra text:
{
  "domain": "{{domain}}",
  "personas": [
    {
      "name": "First persona name",
      "role": "Role or identity (e.g., university student, software engineer, policy advisor)",
      "affiliation": "Affiliation/Organization/Community (if applicable)",
      "background": "Detailed background, including education, work/interest history, and relevant experience (100-150 words)",
      "subdomain": "subdomain within the domain",
    },
    {
      "name": "Second persona name",
      "role": "Role or identity",
      "affiliation": "Affiliation/Organization/Community (if applicable)",
      "background": "Detailed background (100-150 words)",
      "subdomain": "subdomain within the domain",
    },
    ...
  ]
}"""


# =============================================================================
# Deep Research Task Generation Prompt
# =============================================================================

QUERY_GENERATION_PROMPT = """You are a deep research query designer. Based on the persona profile below, generate 4 realistic deep research queries that the persona would like to ask, which MUST rely on deep web search, reasoning, and information synthesis to complete.

Persona information:
- Name: {{name}}
- Role: {{role}}
- Affiliation: {{affiliation}}
- Background: {{background}}
- Subdomain: {{subdomain}}

Query requirements:
1. Each query must require multiple rounds of web search (at least 2 rounds from different perspectives).
2. Each query must integrate information from multiple credible sources (e.g., academic papers, industry reports, news articles, policy documents, statistics, online forums).
3. Cover latest developments, data analysis, trend assessment, comparisons, or case studies — aligned with the persona's role and domain interest.
4. Complexity should be proportional to the persona's profile:
   - For experts: advanced analytical or strategic-level queries.
   - For non-experts: practical, decision-making, or problem-solving queries with substantial information gathering.
5. 70% of queries should include explicit time constraints (e.g., "from January 2024 to August 2025", "before 2025", "as of 2025-08-05").
6. Each query must be concrete, specific, and have clear deliverables.
7. Queries must reflect the persona's realistic needs and challenges.
8. Length: 10-50 words per query.
9. It would be better if query is **tied to a current or emerging hot topic** (e.g., stablecoin regulations in May 2025, AI safety debates, semiconductor export bans) that requires synthesis of conflicting perspectives and time-sensitive evidence.
10. Encourage queries that involve **controversial viewpoints, cross-country comparisons, or forward-looking predictions** (e.g., market impacts, regulatory outcomes, adoption scenarios).

Example queries:
1. My son is about to start his university applications in 2025 for postgraduates but he's still uncertain about both his major and which universities to apply to. Could you help me find the top five universities in each of the five broad subjects from the QS World University Rankings by Subject 2025, and also check their standings in the QS World University Rankings 2025 and the Times Higher Education World University Rankings 2025? And I need the home page of the university's official website, standard application deadline for regular decision as well as the application fee without the fee waiver.
2. Please provide a detailed explanation of the differences and connections between Google's recently released A2A protocol and the MCP protocol. Furthermore, elaborate on the innovative aspects of the A2A protocol and the specific problems it is designed to address.
3. What are the investment philosophies of Duan Yongping, Warren Buffett, and Charlie Munger?
4. How did discussions on stablecoin regulations evolve across US, EU, and Asia in May 2025, and what are the projected implications for financial stability and crypto adoption by 2026?

Strictly output the following JSON only, with no extra text:
{
  "persona_name": "persona_name",
  "tasks": [
    {
      "task_id": "task_1",
      "deep_research_query": "Deep research query (10-50 words)",
      "key_challenges": "Main challenges and why deep search is required",
      "expected_search_rounds": 4,
      "time_sensitivity": true,
      "time_constraint": "Explicit time constraint description"
    },
    {
      "task_id": "task_2",
      "deep_research_query": "Deep research query (10-50 words)",
      "key_challenges": "Main challenges",
      "expected_search_rounds": 3,
      "time_sensitivity": false,
      "time_constraint": null
    },
    ...
  ]
}"""


# =============================================================================
# Deep Research Requirement Evaluation Prompt
# =============================================================================

DEEP_RESEARCH_FILTER_PROMPT = """You are a deep research query analysis expert. Evaluate whether the following task truly requires deep web search and information synthesis to complete.

Query: {{deep_research_query}}
Persona background: {{persona_background}}
Query challenges: {{key_challenges}}

Evaluation criteria:
1. Requires up-to-date information that cannot be accurately answered using pre-2023 knowledge alone.
2. Requires cross-verification and integration of multiple credible sources (e.g., academic papers, news, industry reports, policy documents, statistics).
3. Requires multi-angle, multi-layered deep investigation.
4. Complexity matches the persona's role and capabilities:
   - For experts: advanced analytical or strategic-level complexity.
   - For non-experts: realistic but still substantial research complexity beyond casual searching.

Strictly output the following JSON only, with no extra text:
{
  "needs_deep_research": (true/false),
  "confidence_score": (0~1),
  "reasoning": "Detailed rationale explaining why deep search is needed and what information collection and synthesis are required (100-150 words)",
  "search_complexity": ("High"/"Medium"/"Low"),
  "information_sources_needed": ["academic papers", "news", "technical reports", "market data", "policy documents"],
  "latest_info_required": (true/false),
  "cross_domain_integration": (true/false)
}"""


# =============================================================================
# No-Search Baseline Answer Prompt
# =============================================================================

NO_SEARCH_BASELINE_PROMPT = """Based solely on your existing knowledge, without using any external search tools, answer the following query. Provide the best answer you can.

Persona background: {{persona_background}}
Query: {{deep_research_query}}

Requirements:
1. Provide as detailed and complete an answer as you can, considering the persona's role, knowledge scope, and domain interest.
2. If some information is uncertain or may be outdated, clearly state this.
3. Do not fabricate specific data, dates, or facts you are not confident about.
4. Ensure logical structure and clear organization.
5. Demonstrate appropriate depth of understanding for the persona's role:
   - For experts: in-depth analysis with technical or strategic insights.
   - For non-experts: practical, clear, and well-explained reasoning.
6. If the query involves recent developments, acknowledge any knowledge cutoff limitations.

Please provide your best possible answer:"""


# =============================================================================
# Answer Quality Assessment Prompt
# =============================================================================

QUALITY_ASSESSMENT_PROMPT = """You are an answer quality evaluator for deep research queries. Assess the quality and completeness of the following answer for the given query.

Original query: {{deep_research_query}}
Query challenges: {{key_challenges}}
Persona background: {{persona_background}}
No-search answer: {{baseline_answer}}

Evaluation dimensions:
1. Accuracy: Are statements reliable and free of obvious errors?
2. Completeness: Does it cover the core requirements and key aspects of the query?
3. Depth: Is the analysis sufficiently deep and appropriate for the persona's role?
4. Timeliness: Could information be outdated? Does it acknowledge time limitations?
5. Professionalism: Does it meet the expected standards for the persona's role and demonstrate relevant insight?
6. Structure: Is the answer well-organized and logical?

Special focus:
- If the task involves time requirements such as "latest", "recent", or "current", assess whether the answer meets them.
- If the task requires multi-source synthesis, assess whether a single knowledge source is sufficient.
- Assess whether additional search is truly necessary to provide a better answer, based on the persona's role and context.

Strictly output the following JSON only, with no extra text:
{
  "overall_quality": "low",
  "quality_score": 0.3,
  "completeness_score": 0.4,
  "accuracy_score": 0.7,
  "depth_score": 0.2,
  "timeliness_score": 0.1,
  "accuracy_concerns": "Specific accuracy concerns or issues",
  "missing_aspects": "Important aspects and information that are missing",
  "outdated_info": "Potentially outdated content",
  "requires_search": true,
  "search_necessity_reasoning": "Detailed reasoning for why search is needed and what key information is missing (150-200 words)"
}"""


# =============================================================================
# Utility Functions
# =============================================================================

def render_prompt(template: str, variables: Dict[str, Any]) -> str:
    """
    Render a prompt template by replacing {{variable}} placeholders with actual values.

    Args:
        template: Template string containing {{variable}} placeholders
        variables: Dictionary mapping variable names to values

    Returns:
        Rendered prompt string
    """
    result = template
    for key, value in variables.items():
        placeholder = f"{{{{{key}}}}}"
        result = result.replace(placeholder, str(value))
    return result


if __name__ == "__main__":
    # Test template rendering
    print("=" * 60)
    print("Prompt Template Rendering Test")
    print("=" * 60)

    # Test expert generation prompt
    rendered = render_prompt(PERSONA_GENERATION_PROMPT, {"domain": "Artificial Intelligence"})
    print("\n✓ PERSONA_GENERATION_PROMPT rendered successfully")
    print(f"  Length: {len(rendered)} characters")
    print(f"  Contains domain: {'Artificial Intelligence' in rendered}")

    # Test task generation prompt
    expert_vars = {
        "name": "Dr. Alice Chen",
        "role": "AI Research Scientist",
        "affiliation": "Stanford University",
        "background": "10 years in NLP research",
        "subdomain": "Large Language Models",
    }
    rendered = render_prompt(QUERY_GENERATION_PROMPT, expert_vars)
    print("\n✓ QUERY_GENERATION_PROMPT rendered successfully")
    print(f"  Length: {len(rendered)} characters")
    print(f"  Contains name: {'Dr. Alice Chen' in rendered}")

    # Test filter prompt
    filter_vars = {
        "deep_research_query": "What are the latest LLM benchmarks?",
        "persona_background": "AI researcher with 10 years experience",
        "key_challenges": "Need up-to-date benchmark data",
    }
    rendered = render_prompt(DEEP_RESEARCH_FILTER_PROMPT, filter_vars)
    print("\n✓ DEEP_RESEARCH_FILTER_PROMPT rendered successfully")
    print(f"  Length: {len(rendered)} characters")

    print("\n" + "=" * 60)
    print("All prompt template tests passed!")
