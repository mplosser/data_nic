"""
Download National Information Center (NIC) data from FFIEC.

This script downloads bulk data files from the FFIEC NIC DataDownload page:
https://www.ffiec.gov/npw/FinancialReport/DataDownload

Available datasets:
- Attributes (Active): Current active financial institutions
- Attributes (Closed): Last instance of closed institutions
- Attributes (Branches): Branch office information
- Relationships: Ownership between companies
- Transformations: Mergers, acquisitions, and other changes

Usage:
    # Download all datasets (attempts automated download)
    python 01_download.py

    # Download specific datasets
    python 01_download.py --datasets attributes_active relationships

    # Show manual download instructions (if automated download is blocked)
    python 01_download.py --manual

    # List available datasets
    python 01_download.py --list

Note: The FFIEC website has bot protection. If automated downloads fail,
use --manual to get instructions for downloading files through a browser.
"""

import requests
import argparse
import sys
import time
from pathlib import Path
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed

# Default output directory
DEFAULT_OUTPUT_DIR = 'data/raw'

# Base URL for FFIEC NIC data downloads
BASE_URL = "https://www.ffiec.gov/npw/StaticData/DataDownload"

# Available datasets with their file names
# Format: {key: (filename, description)}
DATASETS = {
    'attributes_active': ('CSV_ATTRIBUTES_ACTIVE.ZIP', 'Active financial institutions'),
    'attributes_closed': ('CSV_ATTRIBUTES_CLOSED.ZIP', 'Closed institutions (last instance)'),
    'attributes_branches': ('CSV_ATTRIBUTES_BRANCHES.ZIP', 'Branch office information'),
    'relationships': ('CSV_RELATIONSHIPS.ZIP', 'Ownership relationships between companies'),
    'transformations': ('CSV_TRANSFORMATIONS.ZIP', 'Mergers, acquisitions, and changes'),
}


def create_session():
    """Create requests session with retry logic and browser-like headers."""
    session = requests.Session()

    retry_strategy = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Use complete browser-like headers
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0',
    })

    return session


def init_session(session):
    """Visit the main download page to establish cookies/session."""
    try:
        # Visit the main download page first
        main_url = "https://www.ffiec.gov/npw/FinancialReport/DataDownload"
        response = session.get(main_url, timeout=30)
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Warning: Could not initialize session: {e}")
        return False


def download_file(session, url, output_path, delay=1.0):
    """Download a file with progress bar and retry logic.

    Args:
        session: Requests session
        url: URL to download from
        output_path: Path to save file
        delay: Delay before download (respectful crawling)

    Returns:
        tuple: (success: bool, message: str)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Respectful delay
    if delay > 0:
        time.sleep(delay)

    try:
        # Add referer header for the download
        headers = {'Referer': 'https://www.ffiec.gov/npw/FinancialReport/DataDownload'}
        response = session.get(url, stream=True, timeout=120, headers=headers)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f:
            if total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True,
                         desc=output_path.name, leave=False) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            size = f.write(chunk)
                            pbar.update(size)
            else:
                # No content-length header
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        return True, f"Downloaded ({file_size_mb:.2f} MB)"

    except requests.exceptions.RequestException as e:
        return False, f"Failed: {e}"


def download_dataset(session, dataset_key, output_dir, delay=1.0):
    """Download a single dataset.

    Args:
        session: Requests session
        dataset_key: Key from DATASETS dict
        output_dir: Output directory
        delay: Delay before download

    Returns:
        tuple: (dataset_key, success, message)
    """
    if dataset_key not in DATASETS:
        return dataset_key, False, f"Unknown dataset: {dataset_key}"

    filename, description = DATASETS[dataset_key]
    url = f"{BASE_URL}/{filename}"
    output_path = Path(output_dir) / filename

    # Skip if already exists
    if output_path.exists():
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        return dataset_key, True, f"Already exists ({file_size_mb:.2f} MB)"

    print(f"[{dataset_key}] Downloading {filename}...")
    success, message = download_file(session, url, output_path, delay=delay)

    return dataset_key, success, message


def list_datasets():
    """Print available datasets."""
    print("\nAvailable datasets:")
    print("-" * 70)
    print(f"{'Key':<25} {'Filename':<30} Description")
    print("-" * 70)
    for key, (filename, description) in DATASETS.items():
        print(f"{key:<25} {filename:<30} {description}")
    print("-" * 70)
    print("\nUse --datasets to download specific datasets, e.g.:")
    print("  python 01_download.py --datasets attributes_active relationships")


def print_manual_instructions(output_dir):
    """Print instructions for manual download."""
    print("\n" + "=" * 80)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 80)
    print("""
The FFIEC website has bot protection that may block automated downloads.
Please download the files manually using your web browser:

1. Visit: https://www.ffiec.gov/npw/FinancialReport/DataDownload

2. Download the CSV format files (recommended):
""")
    for key, (filename, description) in DATASETS.items():
        url = f"{BASE_URL}/{filename}"
        print(f"   - {description}:")
        print(f"     {url}")
        print()

    print(f"""3. Save the downloaded ZIP files to: {output_dir}/

4. After downloading, run the parse script:
   python 03_parse_data.py --input-dir {output_dir} --output-dir data/processed

Alternative: Use the web interface
----------------------------------
1. Go to https://www.ffiec.gov/npw/FinancialReport/DataDownload
2. Select "CSV" format for each dataset type
3. Click the download links on the page
4. Save files to {output_dir}/
""")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Download FFIEC NIC bulk data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all datasets
  python 01_download.py

  # Download specific datasets
  python 01_download.py --datasets attributes_active relationships

  # Download to custom directory
  python 01_download.py --output-dir my_data/raw

  # List available datasets
  python 01_download.py --list

  # Show manual download instructions (if automated download fails)
  python 01_download.py --manual

Data Source:
  FFIEC National Information Center
  https://www.ffiec.gov/npw/FinancialReport/DataDownload

Data Dictionary:
  https://www.ffiec.gov/npw/StaticData/DataDownload/NPW%20Data%20Dictionary.pdf

Note:
  The FFIEC website has bot protection. If automated downloads fail
  with 403 errors, use --manual for browser download instructions.
        """
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=list(DATASETS.keys()),
        default=None,
        help='Specific datasets to download (default: all)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})'
    )

    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay between downloads in seconds (default: 1.0)'
    )

    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Download files in parallel (faster but less polite)'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=3,
        help='Number of parallel workers (default: 3)'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List available datasets and exit'
    )

    parser.add_argument(
        '--manual',
        action='store_true',
        help='Show manual download instructions and exit'
    )

    args = parser.parse_args()

    # List datasets and exit
    if args.list:
        list_datasets()
        return 0

    # Show manual instructions and exit
    if args.manual:
        print_manual_instructions(args.output_dir)
        return 0

    # Determine which datasets to download
    datasets_to_download = args.datasets or list(DATASETS.keys())

    print("=" * 80)
    print("FFIEC NATIONAL INFORMATION CENTER DATA DOWNLOAD")
    print("=" * 80)
    print(f"\nDatasets: {', '.join(datasets_to_download)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Parallel: {args.parallel}")
    if args.parallel:
        print(f"Workers: {args.workers}")
    print(f"Delay between requests: {args.delay}s")
    print("=" * 80 + "\n")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Create session and initialize with cookies
    session = create_session()
    print("Initializing session...")
    init_session(session)

    # Track results
    results = {}

    if args.parallel:
        # Parallel download
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    download_dataset, session, key, args.output_dir, args.delay
                ): key for key in datasets_to_download
            }

            for future in as_completed(futures):
                key, success, message = future.result()
                results[key] = (success, message)
                status = "OK" if success else "FAILED"
                print(f"[{key}] {status}: {message}")
    else:
        # Sequential download
        for key in datasets_to_download:
            key, success, message = download_dataset(
                session, key, args.output_dir, args.delay
            )
            results[key] = (success, message)
            status = "OK" if success else "FAILED"
            print(f"[{key}] {status}: {message}")

    # Summary
    successful = [k for k, (s, _) in results.items() if s]
    failed = [k for k, (s, _) in results.items() if not s]

    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"Successfully downloaded: {len(successful)}/{len(results)} datasets")

    if failed:
        print(f"\nFailed downloads:")
        for key in failed:
            _, message = results[key]
            print(f"  - {key}: {message}")
        print("\nTip: The FFIEC website has bot protection. Try manual download:")
        print("     python 01_download.py --manual")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Parse to parquet:")
    print(f"   python 03_parse_data.py --input-dir {args.output_dir} --output-dir data/processed")
    print("\n2. Summarize data:")
    print("   python 04_summarize.py --input-dir data/processed")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
