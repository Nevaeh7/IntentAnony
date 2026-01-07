"""
Export accuracy and utility metrics from all eval_final_result.json files in folders to CSV
Each JSON file will generate a corresponding CSV file in its folder
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from loguru import logger


# Attribute name mapping: JSON key -> CSV display name
PII_TYPE_MAPPING = {
    'age': 'Age',
    'education': 'Edu',
    'gender': 'Gnd',
    'income': 'Inc',
    'location': 'Loc',
    'married': 'Mar',
    'occupation': 'Occ',
    'pobp': 'PoB'
}

# Metric order
METRIC_ORDER = [
    'Overall',
    'Privacy',
    'Age',
    'Edu',
    'Gnd',
    'Inc',
    'Loc',
    'Mar',
    'Occ',
    'PoB',
    'Utility',
    'Mean',
    'Read',
    'Hall',
    'score',
    'bleu',
    'rouge',
    'llm_judge'
]


def find_eval_result_files(root_dir: str) -> List[Path]:
    """
    Recursively find all eval_final_result.json files
    
    Args:
        root_dir: Root directory path
        
    Returns:
        List of file paths
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {root_dir}")
    
    files = list(root_path.rglob('eval_final_result.json'))
    logger.info(f"Found {len(files)} eval_final_result.json files")
    return files


def extract_metrics_from_json(file_path: str) -> Optional[Tuple[Dict[str, Any], str]]:
    """
    Extract metrics and model name from JSON file
    
    Args:
        file_path: JSON file path
        
    Returns:
        Tuple of (dictionary containing all metrics, model name), returns None if parsing fails
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metrics = {}
        
        # Extract privacy metrics
        privacy_scores = data.get('privacy_scores', {})
        
        # Overall Privacy (using overall_accuracy)
        metrics['Privacy'] = privacy_scores.get('overall_accuracy', None)
        
        # Accuracy for each attribute
        per_pii_accuracy = privacy_scores.get('per_pii_type_accuracy', {})
        for key, display_name in PII_TYPE_MAPPING.items():
            metrics[display_name] = per_pii_accuracy.get(key, None)
        
        # Extract utility metrics
        utility_scores = data.get('utility_scores', {})
        llm_utility = utility_scores.get('llm_utility', {})
        score_utility = utility_scores.get('score_utility', {})
        
        # Utility Mean
        metrics['Mean'] = llm_utility.get('meaning', None)
        # Readability
        metrics['Read'] = llm_utility.get('readability', None)
        # Hallucinations
        metrics['Hall'] = llm_utility.get('hallucinations', None)
        
        # Overall Utility (using mean)
        metrics['Utility'] = llm_utility.get('mean', None)
        
        # Score utility metrics
        metrics['score'] = score_utility.get('mean', None)
        metrics['bleu'] = score_utility.get('bleu', None)
        metrics['rouge'] = score_utility.get('rouge', None)
        metrics['llm_judge'] = score_utility.get('llm_judge', None)
        
        # Overall (temporarily empty, can be calculated as needed)
        metrics['Overall'] = None
        
        # Extract model name
        model_name = data.get('anon_model_name', 'Original')
        
        return metrics, model_name
        
    except Exception as e:
        logger.error(f"Failed to parse JSON file {file_path}: {e}")
        return None


def export_single_file_to_csv(json_file_path: Path, include_overall: bool = False) -> bool:
    """
    Export a single eval_final_result.json file to CSV (saved in the same folder)
    
    Args:
        json_file_path: JSON file path
        include_overall: Whether to include Overall row (currently empty)
        
    Returns:
        Whether export was successful
    """
    # Extract metrics and model name
    result = extract_metrics_from_json(str(json_file_path))
    if result is None:
        return False
    
    metrics, column_name = result
    
    # Prepare DataFrame data
    data_rows = []
    
    # Add metric rows in order
    for metric_name in METRIC_ORDER:
        if metric_name == 'Overall' and not include_overall:
            continue
        
        value = metrics.get(metric_name, None)
        data_rows.append({
            'Metric': metric_name,
            column_name: value if value is not None else ''
        })
    
    # Create DataFrame
    df = pd.DataFrame(data_rows)
    
    # Generate output CSV path (in the same folder as JSON file)
    output_csv_path = json_file_path.parent / 'eval_results.csv'
    
    # Save as CSV
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    
    logger.info(f" CSV saved: {output_csv_path}")
    return True


def export_all_files_to_csv(
    root_dir: str,
    include_overall: bool = False
):
    """
    Export all eval_final_result.json files to CSV (each file generates CSV in its folder)
    
    Args:
        root_dir: Root directory path
        include_overall: Whether to include Overall row (currently empty)
    """
    # Find all files
    files = find_eval_result_files(root_dir)
    
    if not files:
        logger.warning("No eval_final_result.json files found")
        return
    
    success_count = 0
    failed_count = 0
    
    for json_file in files:
        try:
            if export_single_file_to_csv(json_file, include_overall):
                success_count += 1
            else:
                failed_count += 1
        except Exception as e:
            failed_count += 1
            logger.error(f"Failed to process file {json_file}: {e}")
    
    logger.info(
        f"\n{'='*60}\n"
        f"Processing complete: {success_count} successful, {failed_count} failed, {len(files)} total\n"
        f"{'='*60}"
    )


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Export eval_final_result.json to CSV (each file generates CSV in its folder)'
    )
    parser.add_argument(
        '--root_dir',
        type=str,
        default=r'D:\PhD\XCode\PrivacyAnoer\anonymized_results',
        help='Root directory path (default: baselines directory)'
    )
    parser.add_argument(
        '--include_overall',
        action='store_true',
        help='Include Overall row (even if empty)'
    )
    
    args = parser.parse_args()
    
    export_all_files_to_csv(
        root_dir=args.root_dir,
        include_overall=args.include_overall
    )


if __name__ == '__main__':
    main()

