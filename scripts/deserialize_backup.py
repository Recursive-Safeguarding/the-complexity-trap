#!/usr/bin/env python3
"""
Deserialize backup parquet files from HuggingFace repository to recover experimental data.

This script processes two types of parquet files:
1. *metadata.parquet: Single index with column names as file names
2. *instance_data.parquet: Multi-index with run/instance_id structure

Usage:
    python scripts/deserialize_backup.py
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict
import pandas as pd
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_output_directory(output_path: Path) -> None:
    """Create output directory if it doesn't exist."""
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory ready: {output_path}")


def write_file_content(file_path: Path, content: str, is_json: bool = False) -> None:
    """Write content to file, handling JSON formatting if needed."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if is_json and content.strip():
        try:
            # Parse and reformat JSON for better readability
            json_data = json.loads(content)
            content = json.dumps(json_data, indent=2)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON content in {file_path}, writing as-is")
    
    file_path.write_text(content, encoding='utf-8')


def is_json_file(filename: str) -> bool:
    """Check if filename indicates a JSON file."""
    return filename.endswith('.json') or filename == 'pred'


def process_metadata_parquet(parquet_file: Path, output_base: Path) -> None:
    """
    Process metadata.parquet files with single index.
    
    Args:
        parquet_file: Path to the metadata parquet file
        output_base: Base output directory
    """
    logger.info(f"Processing metadata file: {parquet_file}")
    
    try:
        df = pd.read_parquet(parquet_file)
        logger.info(f"Loaded metadata dataframe with shape: {df.shape}")
        
        # The index contains the run name
        for run_name in df.index:
            run_dir = output_base / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each column (file) for this run
            for column_name in df.columns:
                content = df.loc[run_name, column_name]
                if pd.isna(content):
                    logger.warning(f"Skipping null content for {run_name}/{column_name}")
                    continue
                    
                file_path = run_dir / column_name
                write_file_content(file_path, content, is_json_file(column_name))
                logger.debug(f"Written: {file_path}")
                
        logger.info(f"Completed processing metadata file: {parquet_file}")
        
    except Exception as e:
        logger.error(f"Error processing metadata file {parquet_file}: {e}")
        raise


def process_instance_data_parquet(parquet_file: Path, output_base: Path) -> None:
    """
    Process instance_data.parquet files with multi-index (run, instance_id).
    
    Args:
        parquet_file: Path to the instance data parquet file
        output_base: Base output directory
    """
    logger.info(f"Processing instance data file: {parquet_file}")
    
    try:
        df = pd.read_parquet(parquet_file)
        logger.info(f"Loaded instance data dataframe with shape: {df.shape}")
        
        # Multi-index: (run_name, instance_id)
        for (run_name, instance_id) in df.index:
            instance_dir = output_base / run_name / instance_id
            instance_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each column (file extension) for this instance
            for column_name in df.columns:
                content = df.loc[(run_name, instance_id), column_name]
                if pd.isna(content):
                    logger.debug(f"Skipping null content for {run_name}/{instance_id}/{column_name}")
                    continue
                
                # File name is instance_id.column_name
                filename = f"{instance_id}.{column_name}"
                file_path = instance_dir / filename
                write_file_content(file_path, content, is_json_file(column_name))
                logger.debug(f"Written: {file_path}")
                
        logger.info(f"Completed processing instance data file: {parquet_file}")
        
    except Exception as e:
        logger.error(f"Error processing instance data file {parquet_file}: {e}")
        raise


def discover_parquet_files(backup_path: Path) -> Dict[str, list]:
    """
    Discover and categorize parquet files in backup directory.
    
    Args:
        backup_path: Path to backup directory
        
    Returns:
        Dictionary with 'metadata' and 'instance_data' lists of file paths
    """
    parquet_files = list(backup_path.glob("*.parquet"))
    
    metadata_files = []
    instance_data_files = []
    
    for file_path in parquet_files:
        if file_path.name.endswith("-metadata.parquet"):
            metadata_files.append(file_path)
        elif file_path.name.endswith("-instance_data.parquet"):
            instance_data_files.append(file_path)
        else:
            logger.warning(f"Unknown parquet file pattern: {file_path}")
    
    logger.info(f"Found {len(metadata_files)} metadata files and {len(instance_data_files)} instance data files")
    
    return {
        'metadata': sorted(metadata_files),
        'instance_data': sorted(instance_data_files)
    }


def process_single_file(file_info: tuple) -> str:
    """
    Process a single parquet file (metadata or instance data).
    
    Args:
        file_info: Tuple of (file_path, output_path, file_type)
        
    Returns:
        Success message for the processed file
    """
    file_path, output_path, file_type = file_info
    
    try:
        if file_type == "metadata":
            process_metadata_parquet(file_path, output_path)
        elif file_type == "instance_data":
            process_instance_data_parquet(file_path, output_path)
        else:
            raise ValueError(f"Unknown file type: {file_type}")
            
        return f"Successfully processed {file_type}: {file_path.name}"
    except Exception as e:
        error_msg = f"Failed to process {file_type} {file_path.name}: {e}"
        logger.error(error_msg)
        raise Exception(error_msg)


def main() -> None:
    """Main function to orchestrate the deserialization process."""
    parser = argparse.ArgumentParser(description="Deserialize backup parquet files")
    parser.add_argument(
        "--backup-path", 
        type=Path,
        default=Path("/s3/lindenbauer/trajectory-summarization-experiments"),
        help="Path to backup directory containing parquet files"
    )
    parser.add_argument(
        "--output-path",
        type=Path, 
        default=Path("/mnt/shared-fs/lindenbauer/swe-agent-distillation/trajectories/lindenbauer/backup_unpacked"),
        help="Output directory for unpacked data"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of worker threads for parallel processing"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    backup_path = args.backup_path
    output_path = args.output_path
    max_workers = args.max_workers
    
    logger.info(f"Starting deserialization from {backup_path} to {output_path}")
    logger.info(f"Using {max_workers} worker threads for parallel processing")
    
    # Validate backup path exists
    if not backup_path.exists():
        raise FileNotFoundError(f"Backup path does not exist: {backup_path}")
    
    # Setup output directory
    setup_output_directory(output_path)
    
    # Discover parquet files
    parquet_files = discover_parquet_files(backup_path)
    
    total_files = len(parquet_files['metadata']) + len(parquet_files['instance_data'])
    logger.info(f"Processing {total_files} files ({len(parquet_files['metadata'])} metadata, {len(parquet_files['instance_data'])} instance data)")
    
    completed_count = 0
    failed_count = 0
    
    # Phase 1: Process metadata files in parallel (creates run directories)
    logger.info("Phase 1: Processing metadata files in parallel...")
    metadata_tasks = [(metadata_file, output_path, "metadata") for metadata_file in parquet_files['metadata']]
    
    if metadata_tasks:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(process_single_file, task): task for task in metadata_tasks}
            
            with tqdm(total=len(metadata_tasks), desc="Processing metadata files") as pbar:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        logger.debug(result)
                        completed_count += 1
                    except Exception as e:
                        logger.error(f"Metadata task failed: {task[0].name} - {e}")
                        failed_count += 1
                    finally:
                        pbar.update(1)
    
    # Phase 2: Process instance data files in parallel (run directories already exist)
    logger.info("Phase 2: Processing instance data files in parallel...")
    instance_tasks = [(instance_file, output_path, "instance_data") for instance_file in parquet_files['instance_data']]
    
    if instance_tasks:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(process_single_file, task): task for task in instance_tasks}
            
            with tqdm(total=len(instance_tasks), desc="Processing instance data files") as pbar:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        logger.debug(result)
                        completed_count += 1
                    except Exception as e:
                        logger.error(f"Instance data task failed: {task[0].name} - {e}")
                        failed_count += 1
                    finally:
                        pbar.update(1)
    
    logger.info(f"Deserialization completed! Successfully processed: {completed_count}, Failed: {failed_count}")
    
    if failed_count > 0:
        logger.warning(f"Some files failed to process. Check logs for details.")
        exit(1)


if __name__ == "__main__":
    main()