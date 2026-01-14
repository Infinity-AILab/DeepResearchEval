#!/usr/bin/env python3
"""
Example usage script for PointwiseEvaluator

This script demonstrates how to use the PointwiseEvaluator for 
Deep Research Arena evaluations.
"""

import os
import yaml
import logging
import argparse
from deepresearcharena.evaluator.pointwise_evaluator import PointwiseEvaluator


def setup_logging(log_level="INFO", log_file=None):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Deep Research Arena Pointwise Evaluator')
    parser.add_argument('--config', type=str, 
                       default="deepresearcharena/config/pointwise.yaml",
                       help='Configuration file path')
    parser.add_argument('--data_dir', type=str,
                       help='Data directory path (overrides config)')
    parser.add_argument('--models', type=str, nargs='+',
                       help='Models to evaluate (overrides config)')
    parser.add_argument('--output_file', type=str,
                       help='Output file path (overrides config)')
    parser.add_argument('--evaluator_model', type=str,
                       help='Evaluator model name (overrides config)')
    parser.add_argument('--api_type', type=str, choices=['openai', 'openrouter'],
                       help='API type (overrides config)')
    parser.add_argument('--cache_dir', type=str,
                       help='Cache directory (overrides config)')
    parser.add_argument('--max_workers', type=int,
                       help='Maximum number of parallel workers (overrides config)')
    parser.add_argument('--max_queries', type=int,
                       help='Maximum number of queries to evaluate (overrides config)')
    parser.add_argument('--query_ids', type=int, nargs='+',
                       help='Specific query IDs to evaluate (overrides config)')
    parser.add_argument('--selection_method', type=str, choices=['first', 'random', 'specified'],
                       help='Query selection method (overrides config)')
    parser.add_argument('--random_seed', type=int,
                       help='Random seed for query selection (overrides config)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Run in dry-run mode (load data and validate config only)')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.data_dir:
        config['data_dir'] = args.data_dir
    if args.models:
        config['target_models'] = args.models
    if args.output_file:
        config['output']['results_file'] = args.output_file
    if args.evaluator_model:
        config['evaluator_model']['name'] = args.evaluator_model
    if args.api_type:
        config['evaluator_model']['api_type'] = args.api_type
    if args.cache_dir:
        config['evaluation']['cache_dir'] = args.cache_dir
    if args.max_workers:
        config['evaluation']['max_workers'] = args.max_workers
    
    # Override query selection configuration
    if 'query_selection' not in config:
        config['query_selection'] = {}
    
    if args.max_queries is not None:
        config['query_selection']['max_queries'] = args.max_queries
    if args.query_ids is not None:
        config['query_selection']['query_ids'] = args.query_ids
        config['query_selection']['selection_method'] = 'specified'
    if args.selection_method:
        config['query_selection']['selection_method'] = args.selection_method
    if args.random_seed is not None:
        config['query_selection']['random_seed'] = args.random_seed
    
    # Setup logging
    setup_logging(
        log_level=config.get('logging', {}).get('level', 'INFO'),
        log_file=config.get('logging', {}).get('file')
    )
    
    print("="*80)
    print("Deep Research Arena Pointwise Evaluator")
    print("="*80)
    print(f"Data directory: {config['data_dir']}")
    print(f"Target models: {config['target_models']}")
    print(f"Evaluator model: {config['evaluator_model']['name']}")
    print(f"Output file: {config['output']['results_file']}")
    print(f"Cache directory: {config['evaluation']['cache_dir']}")
    
    # Display query selection configuration
    query_config = config.get('query_selection', {})
    if query_config.get('max_queries') or query_config.get('query_ids'):
        print(f"Query selection:")
        if query_config.get('max_queries'):
            print(f"  - Max queries: {query_config['max_queries']}")
        if query_config.get('query_ids'):
            print(f"  - Specific query IDs: {query_config['query_ids']}")
        if query_config.get('selection_method'):
            print(f"  - Selection method: {query_config['selection_method']}")
        if query_config.get('random_seed'):
            print(f"  - Random seed: {query_config['random_seed']}")
    else:
        print("Query selection: All queries will be evaluated")
    
    print("="*80)
    
    # Initialize evaluator
    print("Initializing PointwiseEvaluator...")
    evaluator = PointwiseEvaluator(
        data_dir=config['data_dir'],
        model_name=config['evaluator_model']['name'],
        api_type=config['evaluator_model']['api_type'],
        cache_dir=config['evaluation']['cache_dir']
    )
    
    # Load data
    print("Loading evaluation data...")
    try:
        evaluator.load_data()
        print(f"✅ Successfully loaded {len(evaluator.queries)} queries")
        print(f"✅ Successfully loaded {len(evaluator.model_results)} model result sets")
        
        # Print model details
        for model_name, results in evaluator.model_results.items():
            print(f"   - {model_name}: {len(results)} query results")
            
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return 1
    
    # Dry run mode - just validate and exit
    if args.dry_run:
        print("\n🔍 Dry run mode - validation complete!")
        print("Configuration and data loading successful.")
        
        # Show cache statistics if available
        try:
            cache_stats = evaluator.get_cache_statistics()
            if cache_stats:
                print("\nCache Statistics:")
                for cache_name, size in cache_stats.items():
                    print(f"  {cache_name}: {size} items")
        except:
            pass
            
        return 0
    
    # Filter target models to only those with data
    available_models = list(evaluator.model_results.keys())
    target_models = [m for m in config['target_models'] if m in available_models]
    
    if not target_models:
        print(f"❌ No target models found in data. Available models: {available_models}")
        return 1
    
    print(f"\n🎯 Evaluating models: {target_models}")
    
    # Create output directory
    output_dir = os.path.dirname(config['output']['results_file'])
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Run evaluation
        print("\n🚀 Starting pointwise evaluation...")
        results = evaluator.evaluate_all_queries(
            model_names=target_models,
            query_selection_config=config.get('query_selection'),
            max_workers=config.get('evaluation', {}).get('max_workers', 1)
        )
        
        # Print results if configured
        if config['output'].get('print_results', True):
            evaluator.print_results(results)
        
        # Save results
        print(f"\n💾 Saving results to: {config['output']['results_file']}")
        evaluator.save_results(results, config['output']['results_file'])
        
        # Print cache statistics
        try:
            evaluator.print_cache_info()
        except:
            pass
        
        print("\n✅ Evaluation completed successfully!")
        
        # Print summary
        summary = results.get('summary', {})
        if summary:
            print(f"\n📊 Summary:")
            print(f"   Queries processed: {results.get('selected_query_count', len(results.get('query_results', {})))} / {results.get('total_query_count', 'unknown')} total")
            print(f"   Models evaluated: {len(summary.get('models', {}))}")
            
            # Show selected query IDs if limited
            if results.get('selected_query_ids') and results.get('selected_query_count') != results.get('total_query_count'):
                query_ids = results.get('selected_query_ids', [])
                if len(query_ids) <= 10:
                    print(f"   Selected query IDs: {query_ids}")
                else:
                    print(f"   Selected query IDs: {query_ids[:5]} ... {query_ids[-5:]} (showing first 5 and last 5)")
            
            # Print top model
            if summary.get('models'):
                top_model = max(summary['models'].items(), 
                              key=lambda x: x[1].get('average_total_score', 0))
                print(f"   Top model: {top_model[0]} (avg score: {top_model[1].get('average_total_score', 0):.3f})")

        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
