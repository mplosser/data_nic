"""
Summarize NIC parquet files.

This script scans all NIC parquet files and generates a summary showing:
- Dataset name and description
- Number of rows (entities/relationships/transformations)
- Number of variables
- File sizes
- Date range coverage (where applicable)

Usage:
    # Summarize with default parallelization
    python 04_summarize.py --input-dir data/processed

    # Save summary to CSV
    python 04_summarize.py --input-dir data/processed --output-csv nic_summary.csv

    # Disable parallelization (for low-memory systems)
    python 04_summarize.py --input-dir data/processed --no-parallel
"""

import pandas as pd
import pyarrow.parquet as pq
import argparse
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Dataset descriptions
DATASET_DESCRIPTIONS = {
    'attributes_active': 'Active financial institutions',
    'attributes_closed': 'Closed institutions',
    'attributes_branches': 'Branch offices',
    'relationships': 'Ownership relationships',
    'transformations': 'Mergers and acquisitions',
}


def analyze_file(file_path_str):
    """
    Analyze a single parquet file.

    Args:
        file_path_str: Path to parquet file as string

    Returns:
        Dictionary with file info or None if error
    """
    file_path = Path(file_path_str)

    try:
        # Read parquet file
        df = pd.read_parquet(file_path)

        # Get dataset type from filename
        dataset_type = file_path.stem

        # Get file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        # Get date range if date columns exist
        date_cols = ['D_DT_START', 'D_DT_END', 'D_DT_OPEN', 'D_DT_TRANS', 'DT_EXIST',
                     'DT_END', 'DT_START', 'DT_OPEN', 'DT_TRANS']
        date_range = None
        for col in date_cols:
            if col in df.columns:
                try:
                    dates = pd.to_datetime(df[col], errors='coerce').dropna()
                    if len(dates) > 0:
                        date_range = (dates.min(), dates.max())
                        break
                except Exception:
                    continue

        # Get description
        description = DATASET_DESCRIPTIONS.get(dataset_type, 'Unknown dataset')

        # Count variables with descriptions
        table = pq.read_table(file_path)
        desc_count = sum(
            1 for field in table.schema
            if field.metadata and b'description' in field.metadata
        )

        return {
            'dataset': dataset_type,
            'description': description,
            'rows': len(df),
            'variables': len(df.columns),
            'with_descriptions': desc_count,
            'size_mb': file_size_mb,
            'date_min': date_range[0] if date_range else None,
            'date_max': date_range[1] if date_range else None,
            'file': file_path.name
        }

    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Summarize NIC parquet files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate summary
  python 04_summarize.py --input-dir data/processed

  # Save to CSV
  python 04_summarize.py --input-dir data/processed --output-csv nic_summary.csv

  # Disable parallelization (for low-memory systems)
  python 04_summarize.py --input-dir data/processed --no-parallel
        """
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/processed',
        help='Directory containing parquet files (default: data/processed)'
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
        '--output-csv',
        type=str,
        help='Save summary to CSV file'
    )

    args = parser.parse_args()

    # Setup
    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        print(f"ERROR: Directory does not exist: {input_dir}")
        sys.exit(1)

    # Find parquet files
    parquet_files = sorted(input_dir.glob('*.parquet'))

    if not parquet_files:
        print(f"No parquet files found in {input_dir}")
        print("\nTo extract the data, run:")
        print("  python 03_parse_data.py --input-dir data/raw --output-dir data/processed")
        return 1

    # Determine worker count
    if args.no_parallel:
        workers = 1
    elif args.workers:
        workers = args.workers
    else:
        workers = min(multiprocessing.cpu_count(), len(parquet_files))

    print("=" * 80)
    print("NIC DATA SUMMARY")
    print("=" * 80)
    print(f"Directory: {input_dir}")
    print(f"Files found: {len(parquet_files)}")
    print(f"Parallel workers: {workers}")
    print("=" * 80)

    # Analyze files
    results = []

    if workers == 1:
        # Sequential processing
        print("\nAnalyzing files sequentially...")
        for file_path in parquet_files:
            result = analyze_file(str(file_path))
            if result:
                results.append(result)
                print(f"  Processed {result['dataset']}")

    else:
        # Parallel processing
        print(f"\nProcessing files in parallel with {workers} workers...")

        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_file = {
                executor.submit(analyze_file, str(f)): f
                for f in parquet_files
            }

            for future in as_completed(future_to_file):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        print(f"  Processed {result['dataset']}")

                except Exception as e:
                    print(f"  Error: {e}")

    if not results:
        print("\nNo valid data found")
        return 1

    # Create summary DataFrame
    df_summary = pd.DataFrame(results)
    df_summary = df_summary.sort_values('dataset')

    # Print summary table
    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Dataset':<22} {'Rows':>12} {'Vars':>6} {'Desc':>6} {'Size MB':>9}")
    print("-" * 22 + " " + "-" * 12 + " " + "-" * 6 + " " + "-" * 6 + " " + "-" * 9)

    for _, row in df_summary.iterrows():
        print(f"{row['dataset']:<22} {row['rows']:>12,} "
              f"{row['variables']:>6} {row['with_descriptions']:>6} {row['size_mb']:>9.1f}")

    # Overall statistics
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print(f"Total datasets: {len(df_summary)}")
    print(f"Total rows: {df_summary['rows'].sum():,}")
    print(f"Total size: {df_summary['size_mb'].sum():.1f} MB")

    # Date ranges
    has_dates = df_summary[df_summary['date_min'].notna()]
    if not has_dates.empty:
        print(f"\nDate Coverage:")
        for _, row in has_dates.iterrows():
            date_min = row['date_min'].strftime('%Y-%m-%d') if pd.notna(row['date_min']) else 'N/A'
            date_max = row['date_max'].strftime('%Y-%m-%d') if pd.notna(row['date_max']) else 'N/A'
            print(f"  {row['dataset']}: {date_min} to {date_max}")

    # Dataset descriptions
    print(f"\nDataset Descriptions:")
    for _, row in df_summary.iterrows():
        print(f"  {row['dataset']}: {row['description']}")

    print("=" * 80)

    # Save to CSV if requested
    if args.output_csv:
        # Convert dates to strings for CSV
        df_export = df_summary.copy()
        for col in ['date_min', 'date_max']:
            if col in df_export.columns:
                df_export[col] = df_export[col].apply(
                    lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else ''
                )
        df_export.to_csv(args.output_csv, index=False)
        print(f"\nSummary saved to: {args.output_csv}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
