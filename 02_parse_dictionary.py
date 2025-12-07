"""
Extract variable definitions from the NIC Data Dictionary PDF.

This script attempts to parse the FFIEC NIC Data Dictionary PDF and extract
variable names and descriptions into a CSV file for review and use.

Usage:
    # Parse PDF from URL (downloads first)
    python 02_parse_dictionary.py

    # Parse local PDF file
    python 02_parse_dictionary.py --input path/to/dictionary.pdf

    # Specify output file
    python 02_parse_dictionary.py --output data/variable_dictionary.csv

Note: PDF parsing can be fragile. Review the output CSV for accuracy.
"""

import argparse
import sys
import re
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    print("ERROR: pdfplumber is required. Install with:")
    print("  pip install pdfplumber")
    sys.exit(1)

import pandas as pd
import requests

# Default PDF URL
PDF_URL = "https://www.ffiec.gov/npw/StaticData/DataDownload/NPW%20Data%20Dictionary.pdf"
DEFAULT_OUTPUT = "data/variable_dictionary.csv"
DEFAULT_PDF_PATH = "data/raw/NPW_Data_Dictionary.pdf"


def download_pdf(url, output_path):
    """Download PDF from URL.

    Args:
        url: URL to download from
        output_path: Path to save PDF

    Returns:
        True if successful, False otherwise
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading PDF from {url}...")

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            f.write(response.content)

        print(f"Saved to {output_path}")
        return True

    except Exception as e:
        print(f"ERROR: Failed to download PDF: {e}")
        print("\nManual download instructions:")
        print(f"1. Visit: {url}")
        print(f"2. Save the PDF to: {output_path}")
        return False


def extract_tables_from_pdf(pdf_path):
    """Extract all tables from PDF.

    Args:
        pdf_path: Path to PDF file

    Returns:
        List of DataFrames, one per table found
    """
    tables = []

    print(f"Opening PDF: {pdf_path}")

    with pdfplumber.open(pdf_path) as pdf:
        print(f"PDF has {len(pdf.pages)} pages")

        for i, page in enumerate(pdf.pages):
            page_tables = page.extract_tables()

            if page_tables:
                print(f"  Page {i + 1}: Found {len(page_tables)} table(s)")

                for table in page_tables:
                    if table and len(table) > 1:  # Has header + data
                        # First row is typically header
                        header = [str(c).strip() if c else f'col_{j}'
                                  for j, c in enumerate(table[0])]
                        data = table[1:]

                        df = pd.DataFrame(data, columns=header)
                        tables.append(df)

    return tables


def extract_text_from_pdf(pdf_path):
    """Extract all text from PDF for analysis.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Full text content
    """
    text_parts = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)

    return "\n\n".join(text_parts)


def parse_variable_definitions(text):
    """Parse variable definitions from NIC Data Dictionary text.

    The PDF has a consistent structure:
        COLUMN: VARIABLE_NAME
        DATA TYPE: TYPE [SIZE]
        DESCRIPTION: SHORT_DESCRIPTION
        Long description text...

    Args:
        text: Full text content

    Returns:
        List of dicts with variable, data_type, description, full_description, table
    """
    definitions = []

    # Track current section/table
    current_table = 'UNKNOWN'

    # Split into lines for processing
    lines = text.split('\n')

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Track section headers - only change on SECTION markers or TABLE DETAILS
        # Note: Must check SECTION VI before SECTION V since "VI" contains "V"
        if 'SECTION VI' in line or 'TRANSFORMATIONS TABLE DETAILS' in line:
            current_table = 'TRANSFORMATIONS'
        elif 'SECTION V ' in line or 'RELATIONSHIPS TABLE DETAILS' in line:
            # Use 'SECTION V ' (with space) to avoid matching SECTION VI
            current_table = 'RELATIONSHIPS'
        elif 'SECTION IV' in line or 'ATTRIBUTES TABLE DETAILS' in line:
            current_table = 'ATTRIBUTES'

        # Look for COLUMN: pattern
        if line.startswith('COLUMN:'):
            var_name = line.replace('COLUMN:', '').strip()
            # Handle cases like "COLUMN: AUTH_REG_DIST_FRS (ARDF)"
            if '(' in var_name:
                var_name = var_name.split('(')[0].strip()

            data_type = ''
            short_desc = ''
            full_desc_lines = []
            found_description = False

            # Look ahead for DATA TYPE and DESCRIPTION
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith('COLUMN:'):
                next_line = lines[j].strip()

                if next_line.startswith('DATA TYPE:'):
                    data_type = next_line.replace('DATA TYPE:', '').strip()
                elif next_line.startswith('DESCRIPTION:'):
                    short_desc = next_line.replace('DESCRIPTION:', '').strip()
                    found_description = True
                elif found_description and next_line:
                    # Collect full description lines
                    # Stop at section headers or page markers
                    if any(header in next_line for header in
                           ['SECTION IV', 'SECTION V', 'SECTION VI',
                            'ATTRIBUTES TABLE', 'RELATIONSHIPS TABLE', 'TRANSFORMATIONS TABLE']):
                        break
                    # Skip standalone page numbers
                    if next_line.isdigit() and len(next_line) <= 3:
                        j += 1
                        continue
                    full_desc_lines.append(next_line)

                j += 1

            # Combine description - include value codes
            full_desc = short_desc
            if full_desc_lines:
                desc_text = ' '.join(full_desc_lines)
                # Clean up whitespace
                desc_text = ' '.join(desc_text.split())
                full_desc = f"{short_desc}. {desc_text}" if short_desc else desc_text

            if var_name and len(var_name) >= 2:
                definitions.append({
                    'variable': var_name.upper(),
                    'data_type': data_type,
                    'description': short_desc,
                    'full_description': full_desc,
                    'table': current_table,
                })

            i = j - 1  # Continue from where we left off

        i += 1

    return definitions


def combine_and_clean_tables(tables):
    """Combine extracted tables and clean the data.

    Args:
        tables: List of DataFrames

    Returns:
        Combined DataFrame with variable definitions
    """
    if not tables:
        return pd.DataFrame(columns=['variable', 'description', 'table', 'data_type'])

    all_rows = []

    for i, df in enumerate(tables):
        # Normalize column names
        df.columns = [str(c).upper().strip() for c in df.columns]

        # Try to identify variable name and description columns
        var_col = None
        desc_col = None

        for col in df.columns:
            col_lower = col.lower()
            if any(x in col_lower for x in ['field', 'variable', 'column', 'name', 'element']):
                var_col = col
            elif any(x in col_lower for x in ['description', 'definition', 'desc']):
                desc_col = col

        # If we couldn't identify columns, try first two columns
        if var_col is None and len(df.columns) >= 1:
            var_col = df.columns[0]
        if desc_col is None and len(df.columns) >= 2:
            desc_col = df.columns[1]

        if var_col and desc_col:
            for _, row in df.iterrows():
                var_name = str(row.get(var_col, '')).strip()
                description = str(row.get(desc_col, '')).strip()

                # Filter out headers and empty rows
                if var_name and description and var_name.upper() != var_col:
                    # Check if it looks like a variable name (uppercase, underscores)
                    if re.match(r'^[A-Z][A-Z0-9_]*$', var_name.upper()):
                        all_rows.append({
                            'variable': var_name.upper(),
                            'description': description,
                            'table': f'table_{i + 1}',
                            'data_type': '',
                        })

    result_df = pd.DataFrame(all_rows)

    # Deduplicate by variable name
    if not result_df.empty:
        result_df = result_df.drop_duplicates(subset=['variable'], keep='first')
        result_df = result_df.sort_values('variable').reset_index(drop=True)

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description='Extract variable definitions from NIC Data Dictionary PDF',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and parse PDF
  python 02_parse_dictionary.py

  # Parse local PDF
  python 02_parse_dictionary.py --input data/raw/dictionary.pdf

  # Custom output location
  python 02_parse_dictionary.py --output my_variables.csv

  # Just extract raw text for manual review
  python 02_parse_dictionary.py --text-only --output raw_text.txt

Output:
  CSV file with columns: variable, description, table, data_type
  Review and edit as needed before using with parse.py
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help=f'Path to local PDF file (default: downloads from FFIEC)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=DEFAULT_OUTPUT,
        help=f'Output CSV file path (default: {DEFAULT_OUTPUT})'
    )

    parser.add_argument(
        '--text-only',
        action='store_true',
        help='Just extract raw text (for debugging PDF structure)'
    )

    parser.add_argument(
        '--url',
        type=str,
        default=PDF_URL,
        help='URL to download PDF from'
    )

    args = parser.parse_args()

    # Determine PDF path
    if args.input:
        pdf_path = Path(args.input)
        if not pdf_path.exists():
            print(f"ERROR: PDF file not found: {pdf_path}")
            return 1
    else:
        # Download PDF
        pdf_path = Path(DEFAULT_PDF_PATH)
        if not pdf_path.exists():
            if not download_pdf(args.url, pdf_path):
                return 1

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("NIC DATA DICTIONARY EXTRACTION")
    print("=" * 70)

    if args.text_only:
        # Just extract text
        print("\nExtracting raw text...")
        text = extract_text_from_pdf(pdf_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"\nSaved raw text to: {output_path}")
        print(f"Text length: {len(text):,} characters")
        return 0

    # Try table extraction first
    print("\nExtracting tables from PDF...")
    tables = extract_tables_from_pdf(pdf_path)

    if tables:
        print(f"\nFound {len(tables)} tables, processing...")
        df = combine_and_clean_tables(tables)

        if not df.empty:
            df.to_csv(output_path, index=False)
            print(f"\nSaved {len(df)} variable definitions to: {output_path}")
            print("\nPreview:")
            print(df.head(10).to_string(index=False))
            print("\n" + "=" * 70)
            print("NEXT STEPS")
            print("=" * 70)
            print(f"\n1. Review and edit: {output_path}")
            print("2. The CSV can be used to add metadata to parquet files")
            return 0

    # Use text parsing (the PDF doesn't have proper tables, uses COLUMN: format)
    print("\nExtracting text and parsing COLUMN definitions...")
    text = extract_text_from_pdf(pdf_path)

    definitions = parse_variable_definitions(text)

    if definitions:
        df = pd.DataFrame(definitions)
        # Reorder columns
        df = df[['variable', 'table', 'data_type', 'description', 'full_description']]
        df.to_csv(output_path, index=False)

        print(f"\nSaved {len(df)} variable definitions to: {output_path}")

        # Summary by table
        print("\nVariables by table:")
        for table in df['table'].unique():
            count = len(df[df['table'] == table])
            print(f"  {table}: {count} variables")

        print("\nPreview (first 10):")
        preview_df = df[['variable', 'table', 'data_type', 'description']].head(10)
        print(preview_df.to_string(index=False))
    else:
        print("\nCould not extract structured data from PDF.")
        print("Try using --text-only to extract raw text for manual review.")

        # Save raw text as fallback
        text_output = output_path.with_suffix('.txt')
        with open(text_output, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"\nSaved raw text to: {text_output}")
        return 1

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print(f"\n1. Review the CSV: {output_path}")
    print("2. The 'description' column has short descriptions")
    print("3. The 'full_description' column has detailed descriptions with value codes")
    print("4. This CSV can be used to add metadata to parquet files in parse.py")

    return 0


if __name__ == '__main__':
    sys.exit(main())
