#!/usr/bin/env python3
"""
Script to consolidate experimental data from hyperparameter sweep into parquet files.

This script processes each run directory in the hyperparameter sweep directory,
creating a consolidated DataFrame with multi-index (run, instance_id) for each run.
Each run's data is saved as a separate parquet file.
"""

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from huggingface_hub import HfApi

import pandas as pd
from tqdm import tqdm

# Initialize logger for the script
logger = logging.getLogger("scripts.consolidate_run_data")

# Path to evaluation results directory
EVALUATION_RESULTS_DIR = Path("/mnt/shared-fs/lindenbauer/SWE-bench")


def _setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create console handler with formatting
    handler = logging.StreamHandler()
    handler.setLevel(getattr(logging, log_level.upper()))
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    
    # Add handler to logger if not already present
    if not logger.handlers:
        logger.addHandler(handler)
    
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False


def _setup_worker_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration for worker processes."""
    worker_logger = logging.getLogger("scripts.consolidate_run_data")
    if not worker_logger.handlers:
        _setup_logging(log_level)


def extract_suffix(filename: str) -> str:
    """Extract suffix from filename after splitting on dots."""
    parts = filename.split(".")
    return ".".join(parts[1:]) if len(parts) > 1 else filename


def read_file_content(file_path: Path) -> str:
    """Read file content safely, handling potential encoding issues."""
    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return file_path.read_text(encoding="latin-1")
        except UnicodeDecodeError:
            logger.warning(f"Could not decode file {file_path}, using error handling")
            return file_path.read_text(encoding="utf-8", errors="ignore")


def find_evaluation_file(run_name: str) -> Optional[Path]:
    """Find the evaluation JSON file for a given run name."""
    if not EVALUATION_RESULTS_DIR.exists():
        logger.warning(f"Evaluation results directory {EVALUATION_RESULTS_DIR} does not exist")
        return None
    
    for json_file in EVALUATION_RESULTS_DIR.glob("*.json"):
        if json_file.name.endswith(f".{run_name}.json"):
            return json_file
    
    logger.debug(f"No evaluation file found for run: {run_name}")
    return None


def read_evaluation_file(eval_file_path: Path) -> Optional[Dict[str, str]]:
    """Read evaluation JSON file and return its filename and content."""
    try:
        content = eval_file_path.read_text(encoding="utf-8")
        # Validate it's proper JSON
        json.loads(content)
        return {
            "filename": eval_file_path.name,
            "content": content
        }
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON in evaluation file {eval_file_path}: {e}")
        return {
            "filename": eval_file_path.name,
            "content": f"ERROR: Invalid JSON - {e}"
        }
    except Exception as e:
        logger.warning(f"Failed to read evaluation file {eval_file_path}: {e}")
        return {
            "filename": eval_file_path.name,
            "content": f"ERROR: {e}"
        }


def get_run_metadata(run_dir: Path) -> Dict[str, str]:
    """Extract metadata files directly from the run directory."""
    metadata = {}
    
    # Get all files directly in the run directory (not in subdirectories)
    for file_path in run_dir.iterdir():
        if file_path.is_file() and file_path.name:
            try:
                content = read_file_content(file_path)
                metadata[file_path.name] = content
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")
                metadata[file_path.name] = f"ERROR: {e}"
    
    # Add evaluation results if available
    run_name = run_dir.name
    eval_file_path = find_evaluation_file(run_name)
    if eval_file_path:
        eval_data = read_evaluation_file(eval_file_path)
        if eval_data:
            logger.info(f"Found evaluation file for run {run_name}: {eval_data['filename']}")
            metadata[f"evaluation_{eval_data['filename']}"] = eval_data['content']
    else:
        logger.warning(f"No evaluation file found for run: {run_name}")
    
    return metadata


def get_instance_data(instance_dir: Path) -> Dict[str, str]:
    """Extract data files from an instance directory."""
    instance_data = {}
    
    for file_path in instance_dir.iterdir():
        if file_path.is_file():
            try:
                content = read_file_content(file_path)
                suffix = extract_suffix(file_path.name)
                instance_data[suffix] = content
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")
                suffix = extract_suffix(file_path.name)
                instance_data[suffix] = f"ERROR: {e}"
    
    return instance_data


def collect_run_data(run_dir: Path) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    """Collect all data for a single run directory, separating run metadata from instance data."""
    run_name = run_dir.name
    logger.info(f"Processing run: {run_name}")
    
    # Get metadata files from the run directory
    run_metadata = get_run_metadata(run_dir)
    
    # Get all instance directories
    instance_dirs = [d for d in run_dir.iterdir() if d.is_dir()]
    
    if not instance_dirs:
        logger.warning(f"No instance directories found in {run_name}")
        return run_name, pd.DataFrame(), pd.DataFrame()
    
    # Create run metadata DataFrame (single row)
    run_metadata_data = {**run_metadata, "run": run_name}
    run_metadata_df = pd.DataFrame([run_metadata_data])
    run_metadata_df = run_metadata_df.set_index("run")
    
    # Collect instance data
    instance_data_list = []
    
    for instance_dir in instance_dirs:
        instance_id = instance_dir.name
        instance_data = get_instance_data(instance_dir)
        
        # Instance data with run and instance_id
        row_data = {**instance_data}
        row_data["run"] = run_name
        row_data["instance_id"] = instance_id
        
        instance_data_list.append(row_data)
    
    if not instance_data_list:
        logger.warning(f"No instance data collected for run {run_name}")
        return run_name, run_metadata_df, pd.DataFrame()
    
    # Create instance data DataFrame with multi-index
    instance_data_df = pd.DataFrame(instance_data_list)
    instance_data_df = instance_data_df.set_index(["run", "instance_id"])
    
    logger.info(f"Collected run metadata and data for {len(instance_data_df)} instances in run {run_name}")
    
    return run_name, run_metadata_df, instance_data_df


def save_run_data(run_name: str, run_metadata_df: pd.DataFrame, instance_data_df: pd.DataFrame, output_base_dir: Path) -> None:
    """Save run metadata and instance data to separate parquet files."""
    run_metadata_path = output_base_dir / f"{run_name}-metadata.parquet"
    instance_data_path = output_base_dir / f"{run_name}-instance_data.parquet"
    
    try:
        # Save run metadata
        run_metadata_df.to_parquet(run_metadata_path, engine="pyarrow")
        logger.info(f"Saved run metadata ({len(run_metadata_df)} rows) to {run_metadata_path}")
        
        # Save instance data
        instance_data_df.to_parquet(instance_data_path, engine="pyarrow")
        logger.info(f"Saved instance data ({len(instance_data_df)} rows) to {instance_data_path}")
        
    except Exception as e:
        logger.error(f"Failed to save parquet files in {output_base_dir}: {e}")
        raise


def process_single_run(run_dir: Path, log_level: str = "INFO", output_base_dir: Path = None) -> Optional[str]:
    """Process a single run directory and save its data."""
    # Set up logging for worker process
    _setup_worker_logging(log_level)
    
    try:
        run_name, run_metadata_df, instance_data_df = collect_run_data(run_dir)
        
        if run_metadata_df.empty and instance_data_df.empty:
            logger.warning(f"No data to save for run {run_name}")
            return None
        
        save_run_data(run_dir.name, run_metadata_df, instance_data_df, output_base_dir)
        return run_name
        
    except Exception as e:
        logger.error(f"Error processing run {run_dir.name}: {e}")
        return None


def get_run_directories(base_dirs: List[Path]) -> List[Path]:
    """Get all run directories from multiple base directories."""
    all_run_dirs = []
    for base_dir in base_dirs:
        if not base_dir.exists():
            logger.warning(f"Base directory {base_dir} does not exist, skipping")
            continue
        run_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
        logger.info(f"Found {len(run_dirs)} run directories in {base_dir}")
        all_run_dirs.extend(run_dirs)
    
    logger.info(f"Found {len(all_run_dirs)} total run directories across all base directories")
    return all_run_dirs


def consolidate_data(base_dirs: List[Path], max_workers: int = 10, log_level: str = "INFO", output_base_dir: Path = None) -> None:
    """Consolidate all experimental data using parallel processing."""
    run_dirs = get_run_directories(base_dirs)
    
    if not run_dirs:
        logger.error(f"No run directories found in {base_dirs}")
        return
    
    logger.info(f"Processing {len(run_dirs)} runs with {max_workers} workers")
    
    successful_runs = []
    failed_runs = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_run = {executor.submit(process_single_run, run_dir, log_level, output_base_dir): run_dir 
                        for run_dir in run_dirs}
        
        # Process completed tasks
        for future in tqdm(as_completed(future_to_run), total=len(future_to_run), desc="Processing runs"):
            run_dir = future_to_run[future]
            try:
                result = future.result()
                if result is not None:
                    successful_runs.append(result)
                else:
                    failed_runs.append(run_dir.name)
            except Exception as e:
                logger.error(f"Run {run_dir.name} failed with exception: {e}")
                failed_runs.append(run_dir.name)
    
    # Log summary
    logger.info(f"Processing complete:")
    logger.info(f"  Successfully processed: {len(successful_runs)} runs")
    logger.info(f"  Failed: {len(failed_runs)} runs")
    
    if failed_runs:
        logger.warning(f"Failed runs: {failed_runs}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Consolidate experimental data into parquet files"
    )
    parser.add_argument(
        "--input-base-dirs",
        type=Path,
        nargs='+',
        default=[Path("trajectories/main_experiments")],
        help="Base directories containing run directories"
    )
    parser.add_argument(
        "--output-base-dir",
        type=Path,
        default=Path("/path/to/disk/trajectory-summarization-main-experiments/"),
        help="Base directory to which the resulting parquet files are saved"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=30,
        help="Maximum number of parallel workers"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
        help="Logging level"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="anondl4codeagentcontext/main-experiments",
        help="HuggingFace repo ID"
    )
    
    args = parser.parse_args()
    
    _setup_logging(args.log_level)
    
    # Validate that at least one input directory exists
    valid_dirs = [d for d in args.input_base_dirs if d.exists()]
    if not valid_dirs:
        logger.error(f"None of the specified base directories exist: {args.input_base_dirs}")
        return
    
    if len(valid_dirs) < len(args.input_base_dirs):
        missing_dirs = [d for d in args.input_base_dirs if not d.exists()]
        logger.warning(f"Some base directories do not exist and will be skipped: {missing_dirs}")
    
    logger.info(f"Starting data consolidation from {len(valid_dirs)} base directories: {valid_dirs}")
    consolidate_data(args.input_base_dirs, args.max_workers, args.log_level, args.output_base_dir)
    logger.info("Data extraction complete")
    logger.info(f"Uploading data to HuggingFace Repo: {args.repo_id}")

    api = HfApi()
    api.upload_large_folder(repo_id=args.repo_id, folder_path=args.output_base_dir, repo_type="dataset",
                      allow_patterns=["*.parquet"], num_workers=args.max_workers)

    logger.info(f"Uploaded data to HuggingFace Repo: {args.repo_id}")


if __name__ == "__main__":
    main()
