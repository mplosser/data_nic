# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FFIEC National Information Center (NIC) Data Pipeline - automates downloading, processing, and analyzing bank/financial institution data from the FFIEC.

## Data Source

- **Download URL**: https://www.ffiec.gov/npw/FinancialReport/DataDownload (zipped CSV files)
- **Data Dictionary**: https://www.ffiec.gov/npw/StaticData/DataDownload/NPW%20Data%20Dictionary.pdf
- **Datasets**: Multiple datasets available (Attributes, Relationships, Transformations, etc.) - each gets its own parquet file

## Commands

### Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

### Full Pipeline
```bash
# Download raw data to data/raw/ (manual download required due to bot protection)
python 01_download.py --manual

# Extract variable dictionary from PDF (optional but recommended)
python 02_parse_dictionary.py

# Parse ZIP→parquet with variable metadata
python 03_parse_data.py --input-dir data/raw --output-dir data/processed

# Summarize data coverage
python 04_summarize.py --input-dir data/processed

# Cleanup raw files (optional)
python 05_cleanup.py --raw
```

## Architecture

### Pipeline
```
01_download.py → 02_parse_dictionary.py → 03_parse_data.py → 04_summarize.py → 05_cleanup.py
(raw ZIP)         (var descriptions)       (parquet)          (verification)    (optional)
```

### Directory Structure
```
data/
├── raw/           # Downloaded ZIP files
└── processed/     # Parquet files (one per dataset type)
```

### Key Design Patterns

- **Variable metadata**: Attach descriptions from data dictionary to parquet columns (like Stata variable labels)
- **Parallelization**: Use `ProcessPoolExecutor` for parsing and summarization
- **HTTP resilience**: Sessions with retry logic
- **Dual encoding**: UTF-8 fallback to Latin-1 for CSV files

## Reference Repositories

Pattern follows similar data pipeline repos:
- `data_sod` - FDIC Summary of Deposits
- `data_fry9c` - FR Y-9C bank holding company reports
- `data_call` - Call Reports