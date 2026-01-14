import json
import logging
import random
from typing import Dict, List, Any, Optional
from statistics import mean
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base_evaluator import BaseEvaluator
from deepresearcharena.prompts.pointwise_prompts import WEIGHT_GENERATION_PROMPT, CRITERIA_GENERATION_PROMPT, SCORING_PROMPT, DIMENSION_GENERATION_PROMPT
from .pointwise_core import PointwiseEvaluatorCore

logger = logging.getLogger(__name__)


class PointwiseEvaluator(BaseEvaluator, PointwiseEvaluatorCore):
    """
    Pointwise evaluator for Deep Research Arena
    
    Inherits from BaseEvaluator and adds specific functionality for:
    - Query-specific dimension generation
    - Hierarchical weight calculation
    - Criteria generation for each dimension
    - Pointwise scoring of reports
    """
    
    def __init__(self, 
                 data_dir: str,
                 model_name: str = "gpt-5-mini",
                 api_type: str = "openai",
                 cache_dir: str = None):
        """Initialize the pointwise evaluator"""
        super().__init__(data_dir, model_name, api_type, cache_dir)
        
        # Fixed evaluation dimensions
        self.fixed_dims = {
            "coverage": "Breadth, depth, and relevance of coverage",
            "insight": "Depth, originality, logic, and value of analysis",
            "instruction_following": "Accuracy in meeting all requirements",
            "clarity": "Readability, fluency, structure, and ease of understanding"
        }
        
        # Initialize prompt templates
        self._init_prompts()
        
        logger.info("Initialized PointwiseEvaluator")
    
    def _init_prompts(self):
        """Initialize all prompt templates"""
        # Dimension generation prompt (embedded since it's specific to this class)
        self.dimension_generation_prompt = DIMENSION_GENERATION_PROMPT
        self.weight_generation_prompt = WEIGHT_GENERATION_PROMPT
        self.criteria_generation_prompt = CRITERIA_GENERATION_PROMPT  
        self.scoring_prompt = SCORING_PROMPT

    def _build_all_dims(self, additional_dimensions: List[Dict[str, str]]) -> Dict[str, str]:
        """Build dictionary of all dimensions with their definitions"""
        all_dims_with_definition = self.fixed_dims.copy()
        
        for item in additional_dimensions:
            key = item['meta_dimension_name'].lower().replace(" ", "_").replace("-", "_")
            all_dims_with_definition[key] = item['definition']
        
        return all_dims_with_definition
    
    def select_queries(self, query_selection_config: Dict[str, Any] = None) -> Dict[int, Dict[str, Any]]:
        """
        Select queries based on configuration
        
        Args:
            query_selection_config: Query selection configuration
            
        Returns:
            Dictionary of selected queries
        """
        if query_selection_config is None:
            return self.queries
        
        max_queries = query_selection_config.get('max_queries')
        query_ids = query_selection_config.get('query_ids')
        selection_method = query_selection_config.get('selection_method', 'first')
        random_seed = query_selection_config.get('random_seed', 42)
        
        # If no limits specified, return all queries
        if max_queries is None and query_ids is None:
            logger.info(f"No query selection limits specified, using all {len(self.queries)} queries")
            return self.queries
        
        # If specific query IDs are provided
        if query_ids is not None:
            selected_queries = {}
            for query_id in query_ids:
                if query_id in self.queries:
                    selected_queries[query_id] = self.queries[query_id]
                else:
                    logger.warning(f"Query ID {query_id} not found in loaded queries")
            logger.info(f"Selected {len(selected_queries)} specific queries: {list(selected_queries.keys())}")
            return selected_queries
        
        # If max_queries is specified
        if max_queries is not None and max_queries > 0:
            all_query_ids = list(self.queries.keys())
            
            if max_queries >= len(all_query_ids):
                logger.info(f"Requested {max_queries} queries, but only {len(all_query_ids)} available. Using all queries.")
                return self.queries
            
            # Select queries based on method
            if selection_method == 'first':
                selected_ids = all_query_ids[:max_queries]
                logger.info(f"Selected first {max_queries} queries: {selected_ids}")
            elif selection_method == 'random':
                random.seed(random_seed)
                selected_ids = random.sample(all_query_ids, max_queries)
                selected_ids.sort()  # Sort for consistent ordering
                logger.info(f"Selected {max_queries} random queries (seed={random_seed}): {selected_ids}")
            else:
                logger.warning(f"Unknown selection method '{selection_method}', using 'first'")
                selected_ids = all_query_ids[:max_queries]
                logger.info(f"Selected first {max_queries} queries: {selected_ids}")
            
            selected_queries = {qid: self.queries[qid] for qid in selected_ids}
            return selected_queries
        
        return self.queries

    def create_criteria_for_all_dimensions(self, query_id: int) -> Dict[str, Any]:
        """Generate criteria for all dimensions for a specific query"""
        # Step 1: Generate query-specific dimensions
        additional_dimensions = self.generate_query_dimensions(query_id)
        
        # Step 2: Generate hierarchical weights
        dimension_weights = self.generate_hierarchical_weights(query_id, additional_dimensions)

        all_dims_with_definition = self._build_all_dims(additional_dimensions)

        # Step 3: Generate criteria for all dimensions
        all_criteria = {}
        
        for dim_name in all_dims_with_definition:
            criteria = self.generate_dimension_criteria(query_id, dim_name, all_dims_with_definition)
            all_criteria[dim_name] = criteria

        # Store results for this query
        return {
            'query_id': query_id,
            'all_criteria': all_criteria,
            'all_dims_with_definition': all_dims_with_definition,
            'dimension_weights': dimension_weights,
            'additional_dimensions': additional_dimensions
        }

    def evaluate_query(self, query_id: int, model_names: List[str] = None, 
                      criteria_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evaluate a single query across all specified models"""
        if model_names is None:
            model_names = list(self.model_results.keys())
        
        logger.info(f"Evaluating query {query_id} for models: {model_names}")
    
        # Use provided criteria data or generate new ones
        if criteria_data:
            all_criteria = criteria_data['all_criteria']
            all_dims_with_definition = criteria_data['all_dims_with_definition']
            dimension_weights = criteria_data['dimension_weights']
            additional_dimensions = criteria_data['additional_dimensions']
        else:
            criteria_data = self.create_criteria_for_all_dimensions(query_id)
            all_criteria = criteria_data['all_criteria']
            all_dims_with_definition = criteria_data['all_dims_with_definition']
            dimension_weights = criteria_data['dimension_weights']
            additional_dimensions = criteria_data['additional_dimensions']
        
        # Step 4: Evaluate each model
        query_results = {
            'query_id': query_id,
            'query_prompt': self.queries[query_id]['prompt'],
            "all_dims_with_definition": all_dims_with_definition,
            'additional_dimensions': additional_dimensions,
            'dimension_weights': dimension_weights,
            'all_criteria': all_criteria,
            'model_results': {}
        }
        
        for model_name in model_names:
            if query_id in self.model_results.get(model_name, {}):
                print ("=query_id=: ", query_id)
                report = self.model_results[model_name][query_id]
                
                # Check cache for model results
                cache_key = f"model_result_{query_id}_{model_name}"
                cached_result = self.cache_manager.get("model_results", cache_key)
                
                if cached_result is not None:
                    logger.info(f"Using cached results for query {query_id}, model {model_name}")
                    query_results['model_results'][model_name] = cached_result
                    logger.info(f"Model {model_name} - Total score (cached): {cached_result.get('final_scores', {}).get('total_weighted_score', 0.0):.3f}")
                else:
                    try:
                        # Score the report - use smaller max_workers for dimension-level parallelization
                        # to avoid thread explosion with outer query-level parallelization
                        dimension_max_workers = min(4, len(all_criteria))
                        scores = self.score_report_pointwise(query_id, report, all_criteria, dimension_max_workers)
                        
                        # Calculate final weighted scores
                        final_scores = self.calculate_hierarchical_scores(scores, all_criteria, dimension_weights)
                        
                        result = {
                            'raw_scores': scores,
                            'final_scores': final_scores,
                            'report_text': report
                        }
                        
                        query_results['model_results'][model_name] = result
                        
                        # Only cache the result if scoring was successful
                        self.cache_manager.set("model_results", cache_key, result)
                        
                        logger.info(f"Model {model_name} - Total score: {final_scores.get('total_weighted_score', 0.0):.3f}")
                        
                    except Exception as e:
                        logger.error(f"Failed to score model {model_name} for query {query_id}: {e}")
                        logger.info(f"Skipping cache save for model {model_name}, query {query_id} due to scoring error - can retry later")
                        # Don't add to query_results['model_results'] and don't cache
                        # This allows the query to be retried later
        
        return query_results

    def evaluate_all_queries(self, model_names: List[str] = None, 
                            query_selection_config: Dict[str, Any] = None,
                            max_workers: int = 1) -> Dict[str, Any]:
        """Evaluate all queries across all models"""
        if model_names is None:
            model_names = list(self.model_results.keys())
        
        # Select queries based on configuration
        selected_queries = self.select_queries(query_selection_config)
        
        logger.info(f"Starting evaluation for {len(selected_queries)} selected queries (out of {len(self.queries)} total) and {len(model_names)} models")
        
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'model_names': model_names,
            'query_results': {},
            'summary': {},
            'selected_query_count': len(selected_queries),
            'total_query_count': len(self.queries),
            'selected_query_ids': list(selected_queries.keys())
        }
        
        # Step 1: Parallel criteria generation for selected queries
        logger.info("Generating criteria for selected queries in parallel...")
        query_criteria_map = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all criteria generation tasks for selected queries
            future_to_query = {
                executor.submit(self.create_criteria_for_all_dimensions, query_id): query_id 
                for query_id in selected_queries
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_query):
                query_id = future_to_query[future]
                try:
                    criteria_data = future.result()
                    query_criteria_map[query_id] = criteria_data
                    logger.info(f"Generated criteria for query {query_id}/{len(selected_queries)}")
                except Exception as exc:
                    logger.error(f"Query {query_id} criteria generation failed: {exc}")
        
        # Step 2: Parallel evaluation for selected queries
        logger.info("Evaluating selected queries in parallel...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all evaluation tasks for selected queries
            future_to_query = {
                executor.submit(self.evaluate_query, query_id, model_names, query_criteria_map.get(query_id)): query_id 
                for query_id in selected_queries if query_id in query_criteria_map
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_query):
                query_id = future_to_query[future]
                try:
                    query_result = future.result()
                    all_results['query_results'][query_id] = query_result
                    logger.info(f"Processed query {query_id}/{len(selected_queries)}")
                except Exception as exc:
                    logger.error(f"Query {query_id} evaluation failed: {exc}")

        # Calculate summary statistics
        summary = self._calculate_summary_statistics(all_results['query_results'], model_names)
        all_results['summary'] = summary
        
        return all_results

    def _calculate_summary_statistics(self, query_results: Dict[int, Dict[str, Any]], 
                                    model_names: List[str]) -> Dict[str, Any]:
        """Calculate summary statistics across all queries"""
        summary = {'models': {}}
        
        for model_name in model_names:
            model_scores = []
            dimension_scores = {}
            
            for query_result in query_results.values():
                if model_name in query_result.get('model_results', {}):
                    model_result = query_result['model_results'][model_name]
                    final_scores = model_result.get('final_scores', {})
                    
                    total_score = final_scores.get('total_weighted_score', 0.0)
                    # Only include non-zero scores
                    if total_score > 0.0:
                        model_scores.append(total_score)
                        
                        # Collect dimension scores only for valid evaluations
                        for key, value in final_scores.items():
                            if key.endswith('_score') and key != 'total_weighted_score':
                                if key not in dimension_scores:
                                    dimension_scores[key] = []
                                dimension_scores[key].append(value)
            
            # Calculate averages
            summary['models'][model_name] = {
                'average_total_score': mean(model_scores) if model_scores else 0.0,
                'total_queries': len(model_scores),  # Now only counts valid queries
                'dimension_averages': {
                    dim: mean(scores) if scores else 0.0 
                    for dim, scores in dimension_scores.items()
                }
            }
        
        return summary

    def print_results(self, results: Dict[str, Any]):
        """Print formatted evaluation results"""
        print("\n" + "="*80)
        print("Deep Research Pointwise Evaluation Results")
        print("="*80)
        
        summary = results.get('summary', {})
        model_summaries = summary.get('models', {})
        
        if not model_summaries:
            print("No results to display.")
            return
        
        # Define fixed dimensions
        fixed_dimensions = ['coverage_score', 'insight_score', 'instruction_following_score', 'clarity_score']
        
        # Sort models by average score
        sorted_models = sorted(model_summaries.items(), 
                             key=lambda x: x[1]['average_total_score'], 
                             reverse=True)
        
        # Calculate meta dimensions average for each model
        for model_name, model_data in sorted_models:
            dimension_averages = model_data.get('dimension_averages', {})
            meta_scores = []
            for dim, score in dimension_averages.items():
                if dim.endswith('_score') and dim not in fixed_dimensions:
                    meta_scores.append(score)
            model_data['meta_avg'] = sum(meta_scores) / len(meta_scores) if meta_scores else 0.0
        
        # Print header
        print(f"\n{'Rank':<4} {'Model':<25} {'Avg Score':<10} {'Cove':<6} {'Insight':<7} {'InstrF':<6} {'Clar':<6} {'Meta':<6} {'Queries':<8}")
        print("-" * 94)
        
        # Print data rows
        for rank, (model_name, model_data) in enumerate(sorted_models, 1):
            avg_score = model_data['average_total_score']
            query_count = model_data['total_queries']
            dimension_averages = model_data.get('dimension_averages', {})
            meta_avg = model_data['meta_avg']
            
            # Get fixed dimension scores
            coverage_score = dimension_averages.get('coverage_score', 0.0)
            insight_score = dimension_averages.get('insight_score', 0.0)
            instrf_score = dimension_averages.get('instruction_following_score', 0.0)
            clarity_score = dimension_averages.get('clarity_score', 0.0)
            
            print(f"{rank:<4} {model_name:<25} {avg_score:<10.3f} {coverage_score:<6.1f} {insight_score:<7.1f} {instrf_score:<6.1f} {clarity_score:<6.1f} {meta_avg:<6.1f} {query_count:<8}")
        
        print("\n" + "="*80)
        print("Evaluation completed successfully!")
        print("Total cost: ", self.client._total_cost)
        print("="*80)

