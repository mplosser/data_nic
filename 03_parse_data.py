"""
Extract NIC data from ZIP files to parquet format.

This script processes downloaded NIC files (ZIP) and converts them to
standardized parquet format with parallelization support.

Handles the following datasets:
- CSV_ATTRIBUTES_ACTIVE.ZIP: Active financial institutions
- CSV_ATTRIBUTES_CLOSED.ZIP: Closed institutions
- CSV_ATTRIBUTES_BRANCHES.ZIP: Branch office information
- CSV_RELATIONSHIPS.ZIP: Ownership relationships
- CSV_TRANSFORMATIONS.ZIP: Mergers and acquisitions

Usage:
    # Extract all files with default parallelization
    python 03_parse_data.py --input-dir data/raw --output-dir data/processed

    # Specify number of workers
    python 03_parse_data.py --input-dir data/raw --output-dir data/processed --workers 4

    # Disable parallelization
    python 03_parse_data.py --input-dir data/raw --output-dir data/processed --no-parallel

    # With variable descriptions from data dictionary
    python 03_parse_data.py --dictionary data/variable_dictionary.csv
"""

import io
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zipfile
import argparse
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import re

# Default dictionary path
DEFAULT_DICTIONARY_PATH = Path(__file__).parent / "data" / "variable_dictionary.csv"

# Mapping from ZIP filename to output name and description
DATASET_INFO = {
    'CSV_ATTRIBUTES_ACTIVE': ('attributes_active', 'Active financial institutions'),
    'CSV_ATTRIBUTES_CLOSED': ('attributes_closed', 'Closed institutions'),
    'CSV_ATTRIBUTES_BRANCHES': ('attributes_branches', 'Branch offices'),
    'CSV_RELATIONSHIPS': ('relationships', 'Ownership relationships'),
    'CSV_TRANSFORMATIONS': ('transformations', 'Mergers and acquisitions'),
}

# Map dataset types to table names in dictionary
DATASET_TO_TABLE = {
    'attributes_active': 'ATTRIBUTES',
    'attributes_closed': 'ATTRIBUTES',
    'attributes_branches': 'ATTRIBUTES',
    'relationships': 'RELATIONSHIPS',
    'transformations': 'TRANSFORMATIONS',
}


def load_variable_dictionary(dict_path):
    """
    Load variable descriptions from dictionary CSV.

    Args:
        dict_path: Path to variable_dictionary.csv

    Returns:
        Dict mapping (table, variable) to description
    """
    dict_path = Path(dict_path)
    if not dict_path.exists():
        return {}

    try:
        df = pd.read_csv(dict_path)
        descriptions = {}

        for _, row in df.iterrows():
            var = str(row.get('variable', '')).upper().strip()
            table = str(row.get('table', '')).upper().strip()
            # Prefer full_description if available, fall back to description
            desc = row.get('full_description', '') or row.get('description', '')

            if var and desc:
                descriptions[(table, var)] = str(desc)

        return descriptions

    except Exception as e:
        print(f"Warning: Could not load dictionary: {e}")
        return {}


def create_parquet_schema_with_descriptions(df, descriptions, table_name):
    """
    Create PyArrow schema with field descriptions as metadata.

    Args:
        df: DataFrame to create schema for
        descriptions: Dict mapping (table, variable) to descriptions
        table_name: Table name to look up (ATTRIBUTES, RELATIONSHIPS, TRANSFORMATIONS)

    Returns:
        PyArrow schema with metadata
    """
    fields = []
    for col in df.columns:
        dtype = df[col].dtype

        # Determine PyArrow type based on pandas dtype
        if pd.api.types.is_datetime64_any_dtype(dtype):
            pa_type = pa.timestamp('ns')
        elif pd.api.types.is_integer_dtype(dtype):
            pa_type = pa.int64()
        elif pd.api.types.is_float_dtype(dtype):
            pa_type = pa.float64()
        elif pd.api.types.is_bool_dtype(dtype):
            pa_type = pa.bool_()
        elif dtype == 'object' or pd.api.types.is_string_dtype(dtype):
            # Handle object/string types (numpy type 17)
            pa_type = pa.string()
        else:
            # Fallback: try from_numpy_dtype, default to string if it fails
            try:
                pa_type = pa.from_numpy_dtype(dtype)
            except Exception:
                pa_type = pa.string()

        # Get description for this field
        desc = descriptions.get((table_name, col.upper()), '')

        # Create field with metadata
        if desc:
            field = pa.field(col, pa_type, metadata={'description': desc.encode('utf-8')})
        else:
            field = pa.field(col, pa_type)

        fields.append(field)

    return pa.schema(fields)


def get_dataset_type(filename):
    """
    Extract dataset type from filename.

    Args:
        filename: Name of the ZIP file

    Returns:
        Tuple of (output_name, description) or (None, None) if not recognized
    """
    # Remove extension and match against known patterns
    base_name = Path(filename).stem.upper()

    for pattern, (output_name, description) in DATASET_INFO.items():
        if pattern in base_name:
            return output_name, description

    return None, None


def read_csv_with_encoding(file_handle, **kwargs):
    """
    Read CSV with automatic encoding detection.

    Args:
        file_handle: File handle to read from
        **kwargs: Additional arguments for pd.read_csv

    Returns:
        DataFrame
    """
    # Read content into memory first (ZipExtFile doesn't support seek)
    content = file_handle.read()
    try:
        return pd.read_csv(io.BytesIO(content), encoding='utf-8', low_memory=False, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(content), encoding='latin-1', low_memory=False, **kwargs)


def process_zip_file(zip_path):
    """
    Extract data from NIC ZIP file.

    Each ZIP may contain multiple CSV files. We combine them into one DataFrame.

    Args:
        zip_path: Path to ZIP file

    Returns:
        DataFrame with combined data
    """
    all_dfs = []

    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Find all CSV files in the ZIP
        csv_files = [f for f in zf.namelist() if f.lower().endswith('.csv')]

        if not csv_files:
            raise ValueError(f"No CSV files found in {zip_path}")

        for csv_file in csv_files:
            with zf.open(csv_file) as f:
                df = read_csv_with_encoding(f)
                all_dfs.append(df)

    # Combine all CSVs if multiple
    if len(all_dfs) == 1:
        return all_dfs[0]
    else:
        # Concatenate, assuming same schema
        return pd.concat(all_dfs, ignore_index=True)


def standardize_dataframe(df, dataset_type):
    """
    Standardize NIC DataFrame.

    - Uppercase column names
    - Handle primary identifiers (ID_RSSD for most, varies by dataset)
    - Order columns consistently

    Args:
        df: Raw DataFrame
        dataset_type: Type of dataset (e.g., 'attributes_active')

    Returns:
        Standardized DataFrame
    """
    # Uppercase column names and strip whitespace
    df.columns = [str(col).upper().strip() for col in df.columns]

    # Primary identifier columns by dataset type
    # ID_RSSD is the main entity identifier in NIC data
    id_cols = []

    # Common identifier columns to look for
    potential_id_cols = ['ID_RSSD', 'ID_RSSD_HD_OFF', 'ID_RSSD_PREDECESSOR',
                         'ID_RSSD_SUCCESSOR', 'ID_RSSD_PARENT', 'ID_RSSD_OFFSPRING']

    for col in potential_id_cols:
        if col in df.columns:
            id_cols.append(col)

    # Convert ID columns to integer where possible
    for col in id_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Add date columns to id_cols if present
    date_cols = ['D_DT_START', 'D_DT_END', 'D_DT_OPEN', 'D_DT_TRANS', 'DT_EXIST',
                 'DT_END', 'DT_START', 'DT_OPEN', 'DT_TRANS']
    for col in date_cols:
        if col in df.columns and col not in id_cols:
            id_cols.append(col)

    # Data columns (alphabetical, excluding id columns)
    data_cols = sorted([c for c in df.columns if c not in id_cols])

    # Reorder: ID columns first, then alphabetical data columns
    final_cols = [c for c in id_cols if c in df.columns] + data_cols
    df = df[final_cols]

    return df


def save_parquet_with_metadata(df, output_path, dataset_description, dataset_type, var_descriptions=None):
    """
    Save DataFrame to parquet with custom metadata.

    Args:
        df: DataFrame to save
        output_path: Path to save parquet file
        dataset_description: Description of the dataset
        dataset_type: Type of dataset (e.g., 'attributes_active')
        var_descriptions: Dict mapping (table, variable) to descriptions
    """
    # Get table name for looking up descriptions
    table_name = DATASET_TO_TABLE.get(dataset_type, 'UNKNOWN')

    # Create schema with field descriptions if available
    if var_descriptions:
        schema = create_parquet_schema_with_descriptions(df, var_descriptions, table_name)
        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    else:
        table = pa.Table.from_pandas(df, preserve_index=False)

    # Add custom metadata
    custom_metadata = {
        'dataset_description': dataset_description,
        'source': 'FFIEC National Information Center',
        'source_url': 'https://www.ffiec.gov/npw/FinancialReport/DataDownload',
    }

    # Merge with existing metadata
    existing_metadata = table.schema.metadata or {}
    merged_metadata = {**existing_metadata, **{k.encode(): v.encode() for k, v in custom_metadata.items()}}

    # Create new table with updated metadata
    table = table.replace_schema_metadata(merged_metadata)

    # Write to parquet
    pq.write_table(table, output_path, compression='snappy')


def process_file_wrapper(args_tuple):
    """
    Wrapper function for parallel processing.

    Args:
        args_tuple: (file_path_str, output_dir_str, var_descriptions, force)

    Returns:
        Tuple of (status, dataset_type, message)
    """
    file_path_str, output_dir_str, var_descriptions, force = args_tuple

    file_path = Path(file_path_str)
    output_dir = Path(output_dir_str)

    try:
        # Determine dataset type
        dataset_type, description = get_dataset_type(file_path.name)

        if dataset_type is None:
            return ('error', None, f"Unknown dataset type: {file_path.name}")

        # Output path
        output_path = output_dir / f"{dataset_type}.parquet"

        # Check if already processed
        if output_path.exists() and not force:
            return ('skipped', dataset_type, "Already exists")

        # Process ZIP file
        df = process_zip_file(file_path)

        # Standardize
        df = standardize_dataframe(df, dataset_type)

        # Save as parquet with metadata
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_parquet_with_metadata(df, output_path, description, dataset_type, var_descriptions)

        # Count variables with descriptions
        table_name = DATASET_TO_TABLE.get(dataset_type, 'UNKNOWN')
        desc_count = sum(1 for col in df.columns if (table_name, col.upper()) in (var_descriptions or {}))

        return ('success', dataset_type, f"{len(df):,} rows, {len(df.columns)} columns, {desc_count} descriptions")

    except Exception as e:
        import traceback
        error_msg = f"Error processing {file_path.name}: {str(e)}\n{traceback.format_exc()}"
        return ('error', None, error_msg)


def main():
    parser = argparse.ArgumentParser(
        description='Extract NIC data from ZIP files to parquet format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract with default parallelization (all CPUs)
  python 03_parse_data.py --input-dir data/raw --output-dir data/processed

  # Limit to 4 workers
  python 03_parse_data.py --input-dir data/raw --output-dir data/processed --workers 4

  # Disable parallelization
  python 03_parse_data.py --input-dir data/raw --output-dir data/processed --no-parallel

  # With custom dictionary path
  python 03_parse_data.py --dictionary data/variable_dictionary.csv

  # Skip variable descriptions
  python 03_parse_data.py --no-descriptions

Output Format:
  - Parquet files: {dataset_type}.parquet
  - Columns: ID columns first, then alphabetical
  - Includes dataset and field metadata (variable descriptions)
        """
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/raw',
        help='Directory containing downloaded ZIP files (default: data/raw)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Directory to save parquet files (default: data/processed)'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: all CPUs)'
    )

    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Disable parallel processing'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing parquet files'
    )

    parser.add_argument(
        '--dictionary',
        type=str,
        default=str(DEFAULT_DICTIONARY_PATH),
        help=f'Path to variable dictionary CSV (default: {DEFAULT_DICTIONARY_PATH})'
    )

    parser.add_argument(
        '--no-descriptions',
        action='store_true',
        help='Skip embedding variable descriptions in parquet metadata'
    )

    args = parser.parse_args()

    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find ZIP files to process (deduplicate for case-insensitive filesystems like Windows)
    files_to_process = list(set(input_dir.glob('*.zip')) | set(input_dir.glob('*.ZIP')))
    files_to_process.sort()

    if not files_to_process:
        print(f"No ZIP files found in {input_dir}")
        print("\nTo download the data, run:")
        print("  python 01_download.py")
        print("\nOr for manual download instructions:")
        print("  python 01_download.py --manual")
        return 1

    # Determine worker count
    if args.no_parallel:
        workers = 1
    elif args.workers:
        workers = args.workers
    else:
        workers = min(multiprocessing.cpu_count(), len(files_to_process))

    # Load variable descriptions
    if args.no_descriptions:
        var_descriptions = {}
    else:
        var_descriptions = load_variable_dictionary(args.dictionary)
        if var_descriptions:
            print(f"Loaded {len(var_descriptions)} variable descriptions from dictionary")
        elif Path(args.dictionary).exists():
            print("Warning: Dictionary file exists but could not load descriptions")
        else:
            print(f"Note: No dictionary file found at {args.dictionary}")
            print("  Run: python 02_parse_dictionary.py")

    print("=" * 80)
    print("NIC DATA EXTRACTION")
    print("=" * 80)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Files to process: {len(files_to_process)}")
    print(f"Parallel workers: {workers}")
    print(f"Variable descriptions: {'Yes' if var_descriptions else 'No'}")
    print("=" * 80)

    # Process files
    successful = []
    skipped = []
    failed = []

    if workers == 1:
        # Sequential processing
        print("\nProcessing sequentially...")
        for file_path in files_to_process:
            status, dataset_type, message = process_file_wrapper(
                (str(file_path), str(output_dir), var_descriptions, args.force)
            )

            if status == 'success':
                successful.append(dataset_type)
                print(f"[{dataset_type}] {message}")
            elif status == 'skipped':
                skipped.append(dataset_type)
                print(f"[{dataset_type}] {message}")
            else:
                failed.append(dataset_type if dataset_type else file_path.name)
                print(f"[ERROR] {message}")

    else:
        # Parallel processing
        print(f"\nProcessing in parallel with {workers} workers...")

        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(process_file_wrapper, (str(f), str(output_dir), var_descriptions, args.force)): f
                for f in files_to_process
            }

            # Process results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]

                try:
                    status, dataset_type, message = future.result()

                    if status == 'success':
                        successful.append(dataset_type)
                        print(f"[{dataset_type}] {message}")
                    elif status == 'skipped':
                        skipped.append(dataset_type)
                        print(f"[{dataset_type}] {message}")
                    else:
                        failed.append(dataset_type if dataset_type else file_path.name)
                        print(f"[ERROR] {message}")

                except Exception as e:
                    print(f"[ERROR] Unexpected error processing {file_path.name}: {e}")
                    failed.append(file_path.name)

    # Summary
    print("\n" + "=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"Successfully processed: {len(successful)} datasets")
    if successful:
        print(f"  Datasets: {', '.join(sorted(successful))}")

    if skipped:
        print(f"\nSkipped (already exist): {len(skipped)} datasets")
        print(f"  Datasets: {', '.join(sorted(skipped))}")
        print("  Use --force to reprocess")

    if failed:
        print(f"\nFailed: {len(failed)} datasets")
        print(f"  {failed}")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\nVerify the extracted data:")
    print(f"  python 04_summarize.py --input-dir {output_dir}")

    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
