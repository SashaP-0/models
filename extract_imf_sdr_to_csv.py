import re
from html import unescape
from pathlib import Path

HTML_PATH = Path('/workspace/IMF_report.html')
OUTPUT_CSV_PATH = Path('/workspace/SDR_per_currency_GBP_USD_2000_2015.csv')


def extract_table_rows(html_text: str):
    # Find the specific table that contains the data by looking for the unique title
    # Then extract all tr blocks with class="row1" that have exactly 3 td cells
    rows = []
    # Narrow down to the fancy table area for performance
    table_match = re.search(r'<table[^>]*class="default"[^>]*>([\s\S]*?)</table>', html_text, flags=re.IGNORECASE)
    if not table_match:
        return rows
    table_html = table_match.group(1)

    for tr_html in re.findall(r'<tr\s+class="row1"[^>]*>([\s\S]*?)</tr>', table_html, flags=re.IGNORECASE):
        # Extract td contents
        tds = re.findall(r'<td[^>]*>([\s\S]*?)</td>', tr_html, flags=re.IGNORECASE)
        if len(tds) != 3:
            # Skip title or malformed rows
            continue
        # Clean each cell: unescape HTML entities, strip tags, collapse whitespace
        cleaned = []
        for cell in tds:
            # Remove any HTML tags inside the cell
            cell_no_tags = re.sub(r'<[^>]+>', '', cell)
            cell_unescaped = unescape(cell_no_tags)
            # Normalize non-breaking spaces and stray 'Â' that can appear from mis-decoding
            cell_unescaped = (cell_unescaped
                              .replace('\xa0', ' ')  # U+00A0
                              .replace('\u00a0', ' ')
                              .replace('Â', ''))
            cell_clean = cell_unescaped.strip()
            # Replace multiple spaces with single and strip again
            cell_clean = re.sub(r'\s+', ' ', cell_clean).strip()
            # Some cells may be empty or just non-breaking space -> make truly empty
            if not cell_clean:
                cell_clean = ''
            cleaned.append(cell_clean)
        # Only accept if Date looks like dd-Mmm-yyyy
        if re.match(r'^\d{2}-[A-Za-z]{3}-\d{4}$', cleaned[0]):
            rows.append(tuple(cleaned))
    return rows


def write_csv(rows):
    OUTPUT_CSV_PATH.write_text('', encoding='utf-8')  # ensure overwrite
    with OUTPUT_CSV_PATH.open('w', encoding='utf-8') as f:
        f.write('Date,U.K. pound (GBP),U.S. dollar (USD)\n')
        for date_str, gbp, usd in rows:
            # Keep original date format as in the source
            f.write(f'{date_str},{gbp},{usd}\n')


def read_html_text(path: Path) -> str:
    # Try UTF-8 first; if it fails, fall back to Latin-1
    try:
        return path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        return path.read_text(encoding='latin-1', errors='ignore')


def main():
    if not HTML_PATH.exists():
        raise FileNotFoundError(f'Missing HTML source at {HTML_PATH}')
    html_text = read_html_text(HTML_PATH)
    rows = extract_table_rows(html_text)
    if not rows:
        raise RuntimeError('No data rows extracted from HTML. The page format may have changed.')
    write_csv(rows)
    print(f'Wrote {len(rows)} rows to {OUTPUT_CSV_PATH}')


if __name__ == '__main__':
    main()