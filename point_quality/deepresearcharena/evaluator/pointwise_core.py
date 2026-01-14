import json
import logging
from typing import Dict, List, Any, Optional
from statistics import mean
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class PointwiseEvaluatorCore:
    """Core methods for pointwise evaluation"""

    def generate_query_dimensions(self, query_id: int) -> List[Dict[str, Any]]:
        """Generate query-specific evaluation dimensions"""
        cache_key = f"dimensions_{query_id}"
        cached_result = self.cache_manager.get("dimensions", cache_key)
        
        if cached_result is not None:
            logger.info(f"Using cached dimensions for query {query_id}")
            return cached_result
        
        query_data = self.queries[query_id]
        task_prompt = query_data['prompt']
        
        logger.info(f"Generating query-specific dimensions for query {query_id}")
        
        formatted_prompt = self.dimension_generation_prompt.format(
            task_prompt=task_prompt
        )
        
        messages = [{"role": "user", "content": formatted_prompt}]
        response = self.generate_llm_response(messages, max_tokens=8192, temperature=0.1)

        try:
            dimensions_json = self.extract_json_from_response(response)
            if dimensions_json:
                dimensions = json.loads(dimensions_json)
                logger.info(f"Generated {len(dimensions)} dimensions for query {query_id}")
            else:
                logger.warning(f"Could not extract JSON from response for query {query_id}")
                dimensions = []
        except Exception as e:
            logger.error(f"Failed to parse dimensions for query {query_id}: {e}")
            dimensions = []
        
        # Cache and return results
        self.cache_manager.set("dimensions", cache_key, dimensions)
        return dimensions
    
    def generate_hierarchical_weights(self, query_id: int, additional_dimensions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Generate hierarchical weights for all dimensions"""
        cache_key = f"weights_{query_id}_{len(additional_dimensions)}"
        cached_result = self.cache_manager.get("weights", cache_key)
        
        if cached_result is not None:
            logger.info(f"Using cached weights for query {query_id}")
            return cached_result
        
        query_data = self.queries[query_id]
        task_prompt = query_data['prompt']
        
        logger.info(f"Generating hierarchical weights for query {query_id}")
        
        additional_dimensions_json = json.dumps(additional_dimensions, ensure_ascii=False, indent=2)
        
        formatted_prompt = self.weight_generation_prompt.format(
            task_prompt=task_prompt,
            additional_dimensions_json=additional_dimensions_json
        )
        
        messages = [{"role": "user", "content": formatted_prompt}]
        response = self.generate_llm_response(messages, max_tokens=8192, temperature=0.1)
        # Extract weights from response
        try:
            weights_json = self.extract_json_from_analysis_output(response)
            if weights_json:
                weights = json.loads(weights_json)
                # Normalize weights to sum to 1.0
                total_weight = sum(weights.values())
                if total_weight > 0:
                    weights = {k: v/total_weight for k, v in weights.items()}
                
                # Convert dimension names to lowercase with underscores
                new_weights = {}
                for dim in weights:
                    dim_name = dim.lower().replace(' ', '_').replace('-', '_')
                    new_weights[dim_name] = weights[dim]
                weights = new_weights
                
                logger.info(f"Generated weights for query {query_id}: {weights}")
            else:
                logger.warning(f"Could not extract JSON from weights response for query {query_id}")
                weights = self._get_default_weights(additional_dimensions)
        except Exception as e:
            logger.error(f"Failed to parse weights for query {query_id}: {e}")
            weights = self._get_default_weights(additional_dimensions)
        
        # Cache and return results
        self.cache_manager.set("weights", cache_key, weights)
        return weights
    
    def _get_default_weights(self, additional_dimensions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Get default equal weights if generation failed"""
        num_dims = 4 + len(additional_dimensions)
        equal_weight = 1.0 / num_dims
        
        default_weights = {
            "coverage": equal_weight,
            "insight": equal_weight,
            "instruction_following": equal_weight,
            "clarity": equal_weight
        }
        
        for dim in additional_dimensions:
            dim_name = dim.get('meta_dimension_name', '').lower().replace(' ', '_').replace('-', '_')
            default_weights[dim_name] = equal_weight
            
        return default_weights

    def generate_dimension_criteria(self, query_id: int, dimension_name: str, 
                                  all_dims_with_definition: Dict[str, str]) -> List[Dict[str, Any]]:
        """Generate specific criteria for a dimension"""
        cache_key = f"criteria_{query_id}_{dimension_name}"
        cached_result = self.cache_manager.get("criteria", cache_key)
        
        if cached_result is not None:
            return cached_result
        
        query_data = self.queries[query_id]
        task_prompt = query_data['prompt']
        
        logger.info(f"Generating criteria for dimension '{dimension_name}' in query {query_id}")
        
        # Format meta dimensions string
        meta_dims_str = "\n".join([
            f"- **{dim}**: {all_dims_with_definition[dim]}"
            for dim in all_dims_with_definition
        ])
        
        formatted_prompt = self.criteria_generation_prompt.format(
            task_prompt=task_prompt,
            num_dimensions=len(all_dims_with_definition),
            meta_dimensions=meta_dims_str,
            dimension_name=dimension_name
        )
        
        messages = [{"role": "user", "content": formatted_prompt}]
        response = self.generate_llm_response(messages, max_tokens=8192, temperature=0.1)
        
        try:
            criteria_json = self.extract_json_from_analysis_output(response)
            if criteria_json:
                criteria = json.loads(criteria_json)
                # Normalize criterion weights to sum to 1.0
                if isinstance(criteria, list) and len(criteria) > 0:
                    total_weight = sum(item.get('weight', 0) for item in criteria)
                    if total_weight > 0:
                        for item in criteria:
                            item['weight'] = item.get('weight', 0) / total_weight
                    
                    logger.info(f"Generated {len(criteria)} criteria for dimension '{dimension_name}'")
                else:
                    logger.warning(f"Invalid criteria format for dimension '{dimension_name}'")
                    criteria = self._get_default_criteria(dimension_name)
            else:
                logger.warning(f"Could not extract JSON from criteria response for dimension '{dimension_name}'")
                criteria = self._get_default_criteria(dimension_name)
        except Exception as e:
            logger.error(f"Failed to parse criteria for dimension '{dimension_name}': {e}")
            criteria = self._get_default_criteria(dimension_name)
        
        # Cache and return results
        self.cache_manager.set("criteria", cache_key, criteria)
        return criteria
    
    def _get_default_criteria(self, dimension_name: str) -> List[Dict[str, Any]]:
        """Get default criteria if generation failed"""
        return [{
            "criterion": f"General {dimension_name} assessment",
            "explanation": f"Overall assessment of {dimension_name} quality",
            "weight": 1.0
        }]

    def _score_single_dimension(self, query_id: int, task_prompt: str, report: str, 
                               dim_name: str, criteria_list: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
        """Score a single dimension of a report with retry mechanism"""
        logger.info(f"Scoring dimension: {dim_name}")
        
        # Format criteria for this single dimension
        criteria_for_dimension = [
            {
                "criterion": item["criterion"],
                "explanation": item["explanation"]
            } for item in criteria_list
        ]
        
        # Use the single-dimension scoring prompt
        formatted_prompt = self.scoring_prompt.format(
            task_prompt=task_prompt,
            report=report,
            criteria_of_one_dimension_json=json.dumps(criteria_for_dimension, ensure_ascii=False, indent=2)
        )
        
        messages = [{"role": "user", "content": formatted_prompt}]
        
        # Retry mechanism: try up to 3 times
        max_retries = 3
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                response = self.generate_llm_response(messages, max_tokens=8192, temperature=0.1)

                # Parse the response which should be in format: {"criterion_1": {"analysis": "...", "report_score_0_to_10": x.xx}, ...}
                dimension_response = json.loads(self.extract_json_from_analysis_output(response))

                # Convert to the desired format with criterion text included
                dimension_scores = []
                resp_map = {item["criterion"]: item for item in dimension_response}
                for criterion_item in criteria_list:
                    name = criterion_item["criterion"]
                    item = resp_map[name]
                    dimension_scores.append({
                        "criterion": name,
                        "analysis": item["analysis"],
                        "report_score_0_to_10": float(item["report_score_0_to_10"])
                    })
                
                logger.info(f"Successfully scored dimension '{dim_name}' with {len(dimension_scores)} criteria")
                return dim_name, dimension_scores
                
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for dimension '{dim_name}': {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying dimension '{dim_name}'...")
                continue
        
        # If all retries failed, log error and raise exception to prevent caching
        logger.error(f"All {max_retries} attempts failed for dimension '{dim_name}'. Last error: {last_exception}")
        raise Exception(f"Dimension '{dim_name}' scoring failed after {max_retries} attempts: {last_exception}")

    def score_report_pointwise(self, query_id: int, report: str, all_criteria: Dict[str, List[Dict[str, Any]]], 
                              max_workers: int = None) -> Dict[str, Any]:
        """Score a single report using pointwise evaluation - processes each dimension in parallel"""
        cache_key = f"scores_{query_id}_{hash(report)}"
        cached_result = self.cache_manager.get("scores", cache_key)
        
        if cached_result is not None:
            return cached_result
        
        query_data = self.queries[query_id]
        task_prompt = query_data['prompt']
        
        logger.info(f"Scoring report for query {query_id} - processing {len(all_criteria)} dimensions in parallel")
        
        # Initialize final scores structure
        final_scores = {}
        has_scoring_errors = False
        
        # Use a reasonable default for max_workers if not specified
        # This should be smaller than the outer parallelization to avoid thread explosion
        if max_workers is None:
            max_workers = min(4, len(all_criteria))
        
        # Process dimensions in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all dimension scoring tasks
            future_to_dim = {
                executor.submit(
                    self._score_single_dimension, 
                    query_id, task_prompt, report, dim_name, criteria_list
                ): dim_name 
                for dim_name, criteria_list in all_criteria.items()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_dim):
                dim_name = future_to_dim[future]
                try:
                    result_dim_name, dimension_scores = future.result()
                    final_scores[result_dim_name] = dimension_scores
                except Exception as exc:
                    logger.error(f"Dimension {dim_name} scoring failed: {exc}")
                    # Set empty result for failed dimension to avoid KeyError later
                    final_scores[dim_name] = []
                    has_scoring_errors = True
        
        logger.info(f"Completed scoring for query {query_id} across all dimensions")
        
        # Only cache results if all dimensions scored successfully
        if not has_scoring_errors:
            self.cache_manager.set("scores", cache_key, final_scores)
            logger.info(f"Cached successful scores for query {query_id}")
        else:
            logger.info(f"Skipping cache save for query {query_id} due to dimension scoring errors - can retry later")
        
        return final_scores

    def calculate_hierarchical_scores(self, scores: Dict[str, Any], 
                                    all_criteria: Dict[str, List[Dict[str, Any]]],
                                    dimension_weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate final hierarchical weighted scores"""
        final_scores = {}
        total_weighted_score = 0.0
        
        for dim_name, criteria_list in all_criteria.items():
            if dim_name not in scores:
                continue
                
            # Calculate weighted average for this dimension
            dim_scores = scores[dim_name]
            if not isinstance(dim_scores, list):
                continue
                
            weighted_dim_score = 0.0
            total_criterion_weight = 0.0

            for i, criterion_data in enumerate(criteria_list):
                if i < len(dim_scores):
                    score_item = dim_scores[i]
                    # print ("-------score_item report_score_0_to_10: ", score_item['report_score_0_to_10'])
                    # print ("-------criterion_data weight: ", criterion_data['weight'])
                    if (isinstance(score_item, dict) and 
                        criterion_data['criterion'] == score_item['criterion'] and
                        'report_score_0_to_10' in score_item):
                        
                        score_value = score_item['report_score_0_to_10']
                        criterion_weight = criterion_data['weight']
                        
                        weighted_dim_score += float(score_value) * float(criterion_weight)
                        total_criterion_weight += float(criterion_weight)
            
            if float(total_criterion_weight) > 0:
                final_dim_score = float(weighted_dim_score) / float(total_criterion_weight)
            else:
                final_dim_score = 0.0
                
            final_scores[f"{dim_name}_score"] = final_dim_score
            
            # Add to total weighted score
            dim_weight = dimension_weights[dim_name]
            total_weighted_score += float(final_dim_score) * float(dim_weight)

        final_scores['total_weighted_score'] = float(total_weighted_score)
        return final_scores
