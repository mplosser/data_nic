# FFIEC National Information Center Data Pipeline

Automated pipeline for downloading and processing FFIEC National Information Center (NIC) data.

## Overview

Downloads NIC data from the FFIEC and converts it to parquet format with variable descriptions.

**Datasets:**
- **Attributes (Active)**: Active financial institutions
- **Attributes (Closed)**: Closed institutions
- **Attributes (Branches)**: Branch office information
- **Relationships**: Ownership relationships between entities
- **Transformations**: Mergers and acquisitions

## Requirements

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Download Data

```bash
# Show manual download instructions (automated download blocked by FFIEC)
python 01_download.py --manual

# List available datasets
python 01_download.py --list
```

**Manual Download:**
1. Visit https://www.ffiec.gov/npw/FinancialReport/DataDownload
2. Download the desired ZIP files to `data/raw/`

### 2. Extract Variable Dictionary (Optional)

```bash
# Download and parse data dictionary PDF
python 02_parse_dictionary.py

# Or from local PDF
python 02_parse_dictionary.py --input data/raw/NPW_Data_Dictionary.pdf
```

### 3. Parse to Parquet

```bash
# Extract all files with default parallelization
python 03_parse_data.py --input-dir data/raw --output-dir data/processed

# With custom dictionary path
python 03_parse_data.py --dictionary data/variable_dictionary.csv

# Limit workers for low-memory systems
python 03_parse_data.py --workers 4

# Force reprocessing of existing files
python 03_parse_data.py --force
```

### 4. Verify Data

```bash
# Generate summary
python 04_summarize.py --input-dir data/processed

# Save summary to CSV
python 04_summarize.py --output-csv nic_summary.csv
```

### 5. Cleanup (Optional)

```bash
# Preview what would be deleted
python 05_cleanup.py --raw --dry-run

# Delete raw files (ZIP)
python 05_cleanup.py --raw

# Delete everything (raw + processed)
python 05_cleanup.py --all
```

## Data Sources

| Dataset | Filename | Description |
|---------|----------|-------------|
| Attributes (Active) | CSV_ATTRIBUTES_ACTIVE.ZIP | Currently active financial institutions |
| Attributes (Closed) | CSV_ATTRIBUTES_CLOSED.ZIP | Closed financial institutions |
| Attributes (Branches) | CSV_ATTRIBUTES_BRANCHES.ZIP | Branch office locations |
| Relationships | CSV_RELATIONSHIPS.ZIP | Ownership/control relationships |
| Transformations | CSV_TRANSFORMATIONS.ZIP | Mergers, acquisitions, failures |

**Source:** https://www.ffiec.gov/npw/FinancialReport/DataDownload

## Output Format

All data is saved as parquet files in `data/processed/`:

```
data/processed/
├── attributes_active.parquet
├── attributes_closed.parquet
├── attributes_branches.parquet
├── relationships.parquet
└── transformations.parquet
```

**File Structure:**
- **ID columns first**: ID_RSSD (primary identifier), date columns
- **Data columns**: Alphabetical order
- **Variable descriptions**: Embedded in parquet metadata (like Stata variable labels)

**Variable Descriptions:**

Parquet files include embedded variable descriptions extracted from the NIC Data Dictionary PDF.

```python
import pyarrow.parquet as pq

table = pq.read_table("data/processed/attributes_active.parquet")
for field in table.schema:
    desc = field.metadata.get(b'description', b'').decode() if field.metadata else ''
    print(f"{field.name}: {desc}")
```

## Pipeline Scripts

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `01_download.py` | Download instructions | - | Manual download guide |
| `02_parse_dictionary.py` | Extract variable definitions | PDF | variable_dictionary.csv |
| `03_parse_data.py` | Convert to parquet | ZIP files | Parquet files |
| `04_summarize.py` | Verify data | Parquet files | Summary table |
| `05_cleanup.py` | Delete data files | - | - |

**Parallelization** (03_parse_data.py and 04_summarize.py):
- Default: Uses all CPU cores
- `--workers N`: Limit to N workers
- `--no-parallel`: Sequential processing

## Key Identifiers

- **ID_RSSD**: Primary identifier for financial institutions in NIC data
- **ID_RSSD_HD_OFF**: Head office RSSD (for branches)
- **ID_RSSD_PREDECESSOR/SUCCESSOR**: For transformations
- **ID_RSSD_PARENT/OFFSPRING**: For relationships

## Additional Resources

- **NIC Overview**: https://www.ffiec.gov/NPW.htm
- **Data Download**: https://www.ffiec.gov/npw/FinancialReport/DataDownload
- **Data Dictionary PDF**: https://www.ffiec.gov/npw/StaticData/DataDownload/NPW%20Data%20Dictionary.pdf
