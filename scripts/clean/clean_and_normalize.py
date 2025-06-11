#!/usr/bin/env python3

import pandas as pd
import string
from rapidfuzz import fuzz


def normalize_text(s: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace"""
    return (
        s.lower()
         .translate(str.maketrans('', '', string.punctuation))
         .strip()
         .replace('\n', ' ')
    )


def load_papers(input_path: str) -> pd.DataFrame:
    """Load JSON or JSONL into a DataFrame"""
    try:
        return pd.read_json(input_path, lines=True)
    except ValueError:
        return pd.read_json(input_path)


def flatten_list_of_dicts(series, key='display_name'):
    """Extract list of display_name and normalize each"""
    def _flatten(lst):
        if not isinstance(lst, list):
            return ''
        names = [ normalize_text(item.get(key, '')) for item in lst if isinstance(item, dict) and item.get(key) ]
        return ';'.join(sorted(set(names)))
    return series.apply(_flatten)


def normalize_and_dedupe_papers(
    input_path: str = '../../data/papers-data/cleaned_papers.jsonl',
    output_path: str = '../../data/papers-data/papers_normalized.csv',
    fuzzy_threshold: int = 85
):
    df = load_papers(input_path)

        # Keep only fields needed for linking
    needed = ['id','doi','title','publication_date','keywords','topics','subfields','fields','abstract']
    present = [c for c in needed if c in df.columns]
    df = df[present].copy()
    # Ensure abstract exists
    if 'abstract' not in df.columns:
        df['abstract'] = ''

    needed = ['id','doi','title','publication_date','keywords','topics','subfields','fields','abstract']
    df = df[[c for c in needed if c in df.columns]].copy()

    # Flatten nested lists to semicolon-separated strings
    for col in ['keywords','topics','subfields','fields']:
        df[f'{col}_flat'] = flatten_list_of_dicts(df[col])
        df.drop(columns=[col], inplace=True)

    # Normalize publication_date
    if 'publication_date' in df:
        df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce').dt.strftime('%Y-%m-%d')

    # Normalize text for title and doi
    df['title_norm'] = df['title'].fillna('').map(normalize_text)
    df['doi_norm']   = df['doi'].fillna('').map(lambda s: s.lower().strip())

    # Drop records missing both title and doi
    df = df[(df['title_norm'] != '') | (df['doi_norm'] != '')]

    # Exact dedupe on DOI
    has_doi = df[df['doi_norm'] != ''].drop_duplicates(subset=['doi_norm'], keep='first')
    no_doi  = df[df['doi_norm'] == '']

    # Fuzzy dedupe on title_norm
    titles = no_doi['title_norm'].tolist()
    keep_idx = []
    for idx, t in zip(no_doi.index, titles):
        if any(fuzz.token_sort_ratio(t, titles[j]) >= fuzzy_threshold for j in keep_idx):
            continue
        keep_idx.append(no_doi.index.get_loc(idx))
    dedup_no_doi = no_doi.iloc[keep_idx]

    normalized = pd.concat([has_doi, dedup_no_doi], ignore_index=True)

    # Save flattened CSV
    cols_to_write = [
        'id','doi','doi_norm','title','title_norm','publication_date','abstract',
        'keywords_flat','topics_flat','subfields_flat','fields_flat'
    ]
    normalized.to_csv(output_path, columns=cols_to_write, index=False)
    print(f"Papers normalized: {len(df)} → {len(normalized)} records.")


def normalize_and_dedupe_technologies(
    input_path: str = '../../data/technologies-data/finished_technologies.csv',
    output_path: str = '../../data/technologies-data/technologies_normalized.csv',
    fuzzy_threshold: int = 90
):
    df = pd.read_csv(input_path, header=0, names=['Technology Name'])
    df['tech_norm'] = df['Technology Name'].map(normalize_text)

    # Exact dedupe
    exact = df.drop_duplicates(subset=['tech_norm'], keep='first')

    # Fuzzy dedupe
    names = exact['tech_norm'].tolist()
    keep_idx = []
    for idx, n in zip(exact.index, names):
        if any(fuzz.token_sort_ratio(n, names[j]) >= fuzzy_threshold for j in keep_idx):
            continue
        keep_idx.append(exact.index.get_loc(idx))
    normalized = exact.iloc[keep_idx]

    # Save
    normalized[['Technology Name']].to_csv(output_path, index=False)
    print(f"Technologies normalized: {len(df)} → {len(normalized)} records.")


def main():
    normalize_and_dedupe_papers()
    normalize_and_dedupe_technologies()


if __name__ == '__main__':
    main()
