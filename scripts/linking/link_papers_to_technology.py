import pandas as pd
import string
from rapidfuzz import fuzz
import argparse


def normalize_text(text: str) -> str:
    """Return lowercase text without punctuation and collapsed whitespace."""
    if not isinstance(text, str):
        return ""
    return (
        text.lower()
        .translate(str.maketrans('', '', string.punctuation))
        .replace('\n', ' ')
        .strip()
    )


def link_papers_to_technologies(
    papers_csv: str = '../../data/papers-data/papers_classified.csv',
    tech_csv: str = '../../data/technologies-data/technologies_normalized.csv',
    output_csv: str = '../../data/papers-data/paper_technology_links.csv',
    fuzzy_threshold: int = 85,
) -> None:
    """Generate paper-technology links using keyword matching."""
    papers = pd.read_csv(papers_csv)
    techs = pd.read_csv(tech_csv)

    techs['tech_norm'] = techs['Technology Name'].map(normalize_text)
    techs = techs[techs['tech_norm'] != '']

    links = []
    for _, row in papers.iterrows():
        text_fields = [
            row.get('keywords_flat', ''),
            row.get('topics_flat', ''),
            row.get('subfields_flat', ''),
            row.get('fields_flat', ''),
            row.get('title', ''),
            row.get('abstract', ''),
        ]
        combined = normalize_text(' '.join(map(str, text_fields)))

        matched = False
        for _, tech_row in techs.iterrows():
            tech_norm = tech_row['tech_norm']
            if not tech_norm:
                continue
            if tech_norm in combined or fuzz.partial_ratio(tech_norm, combined) >= fuzzy_threshold:
                links.append({'paper_id': row['id'], 'technology_name': tech_row['Technology Name']})
                matched = True

        if not matched:
            pred = row.get('predicted_technology')
            if isinstance(pred, str) and pred.strip():
                links.append({'paper_id': row['id'], 'technology_name': pred})
    if links:
        pd.DataFrame(links).drop_duplicates().to_csv(output_csv, index=False)
    else:
        pd.DataFrame(columns=['paper_id','technology_name']).to_csv(output_csv, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Link papers to technologies")
    parser.add_argument("--papers", default='../../data/papers-data/papers_classified.csv', help="Input classified papers CSV")
    parser.add_argument("--techs", default='../../data/technologies-data/technologies_normalized.csv', help="Input technologies CSV")
    parser.add_argument("--output", default='../../data/papers-data/paper_technology_links.csv', help="Output links CSV")
    parser.add_argument("--fuzzy-threshold", type=int, default=85, help="Fuzzy matching threshold")
    args = parser.parse_args()
    link_papers_to_technologies(
        papers_csv=args.papers,
        tech_csv=args.techs,
        output_csv=args.output,
        fuzzy_threshold=args.fuzzy_threshold
    )