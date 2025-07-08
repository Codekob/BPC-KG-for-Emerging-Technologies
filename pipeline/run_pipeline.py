import os
import json
import csv
import argparse
import subprocess
from pathlib import Path
import sys

import requests
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
MAIL = os.getenv("MAIL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CRUNCHBASE_API_KEY = os.getenv("CRUNCHBASE_API_KEY")


def enrich_technologies(tech_csv: str) -> Path:
    """Add one-sentence definitions to technologies using OpenAI."""
    df = pd.read_csv(tech_csv, encoding="latin-1")
    df.columns = ["Technology Name"]
    client = OpenAI(api_key=OPENAI_API_KEY)

    def fetch_definition(name: str) -> str:
        prompt = (
            f"Give a one-sentence definition of {name} in lay language. "
            "Only return the one sentence definition. If you don't know, "
            "return 'I don't know'."
        )
        try:
            resp = client.responses.create(model="gpt-4.1-mini", input=prompt)
            return resp.output_text
        except Exception as exc:
            print(f"Error fetching definition for {name}: {exc}")
            return ""

    df["Definition"] = df[df.columns[0]].apply(fetch_definition)
    out_path = Path(__file__).with_name("technologies_with_definitions.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved definitions to {out_path}")
    return out_path


def fetch_and_clean_papers(start_year: int, end_year: int, data_dir: Path) -> Path:
    """Fetch papers from OpenAlex and clean them."""
    data_dir.mkdir(parents=True, exist_ok=True)
    url = "https://api.openalex.org/works"
    years = list(range(end_year, start_year - 1, -1))
    all_papers = []

    for year in years:
        paper_count = 0
        page = 1
        while paper_count < 1000:
            params = {
                "filter": f"publication_year:{year}",
                "mailto": MAIL,
                "sort": "cited_by_count:desc",
                "page": page,
                "per-page": 200,
                "select": (
                    "id, doi, title, publication_date, authorships, cited_by_count, "
                    "keywords, topics, abstract_inverted_index"
                ),
            }
            resp = requests.get(url, params=params)
            if resp.status_code != 200:
                print(f"Error fetching data for year {year}: {resp.status_code}")
                break
            data = resp.json()
            papers = data.get("results", [])
            if not papers:
                break
            papers_with_abs = [p for p in papers if p.get("abstract_inverted_index")]
            paper_count += len(papers_with_abs)
            all_papers.extend(papers_with_abs)
            page += 1

    print(f"Fetched {len(all_papers)} papers")
    raw_path = data_dir / "raw_papers.jsonl"
    with open(raw_path, "w", encoding="utf-8") as fh:
        for p in all_papers:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Raw papers saved to {raw_path}")

    def inv_index(idx: dict) -> str:
        pos2word = {pos: word for word, poses in idx.items() for pos in poses}
        return " ".join(pos2word[i] for i in sorted(pos2word))

    def clean_paper(paper: dict) -> dict:
        paper = paper.copy()
        cleaned_authors = []
        institutions = {}
        for author in paper.get("authorships", []):
            cleaned_authors.append({
                "author_position": author.get("author_position"),
                "display_name": author.get("author", {}).get("display_name"),
                "orcid": author.get("author", {}).get("orcid"),
            })
            for inst in author.get("institutions", []):
                inst = {k: v for k, v in inst.items() if k != "lineage"}
                institutions[inst["id"]] = inst
        paper["authorships"] = cleaned_authors
        paper["institutions"] = list(institutions.values())
        paper["keywords"] = [
            {"id": k.get("id"), "display_name": k.get("display_name"), "score": k.get("score")}
            for k in paper.get("keywords", [])
        ]
        topics_raw = paper.get("topics", [])
        paper["topics"] = [
            {"display_name": t.get("display_name"), "score": t.get("score")}
            for t in topics_raw
        ]
        subfields, fields, domains = {}, {}, {}
        for t in topics_raw:
            sf = t.get("subfield")
            if sf:
                subfields[sf["id"]] = {"id": sf["id"], "display_name": sf["display_name"]}
            field = t.get("field")
            if field:
                fields[field["id"]] = {"id": field["id"], "display_name": field["display_name"]}
            domain = t.get("domain")
            if domain:
                domains[domain["id"]] = {"id": domain["id"], "display_name": domain["display_name"]}
        paper["subfields"] = list(subfields.values())
        paper["fields"] = list(fields.values())
        paper["domains"] = list(domains.values())
        idx = paper.get("abstract_inverted_index")
        if isinstance(idx, dict):
            paper["abstract"] = inv_index(idx)
            paper.pop("abstract_inverted_index", None)
        return paper

    def stream_jsonl(path: Path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)

    def write_jsonl(records, path: Path):
        with open(path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    cleaned_path = data_dir / "cleaned_papers.jsonl"
    cleaned_gen = (clean_paper(p) for p in stream_jsonl(raw_path))
    write_jsonl(cleaned_gen, cleaned_path)
    print(f"Cleaned papers saved to {cleaned_path}")
    return cleaned_path


def fetch_crunchbase_companies(data_dir: Path) -> None:
    """Fetch companies from Crunchbase and save small CSV."""
    data_dir.mkdir(parents=True, exist_ok=True)
    base_url = "https://api.crunchbase.com/v4/data/searches/organizations"
    params = {"user_key": CRUNCHBASE_API_KEY}
    headers = {"Content-Type": "application/json"}
    payload = {
        "field_ids": [
            "identifier",
            "short_description",
            "description",
            "website",
            "company_type",
            "location_identifiers",
            "funding_total",
            "founded_on",
            "funding_stage",
            "category_groups",
            "categories",
        ],
        "query": [
            {"type": "predicate", "field_id": "founded_on", "operator_id": "gte", "values": ["2020-01-01"]},
            {"operator_id": "includes", "type": "predicate", "field_id": "funding_stage", "values": ["seed", "early_stage_venture", "late_stage_venture"]},
            {"type": "predicate", "field_id": "location_identifiers", "operator_id": "includes", "values": ["6106f5dc-823e-5da8-40d7-51612c0b2c4e"]},
        ],
        "order": [{"field_id": "equity_funding_total", "sort": "desc"}],
        "limit": 200,
    }
    resp = requests.post(base_url, params=params, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json().get("entities", [])

    raw_json = data_dir / "crunchbase_fetching_raw.json"
    with open(raw_json, "w") as fh:
        json.dump(data, fh, indent=2)

    csv_path = data_dir / "crunchbase_fetch_linking.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["identifier.value", "identifier.permalink", "description"])
        for entity in data:
            props = entity.get("properties", {})
            identifier = props.get("identifier", {})
            writer.writerow([
                identifier.get("value", ""),
                identifier.get("permalink", ""),
                props.get("description", ""),
            ])
    print(f"Saved Crunchbase data to {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the data pipeline")
    parser.add_argument("--tech-csv", required=True, help="CSV with technologies")
    parser.add_argument("--start-year", type=int, required=True, help="Start year for papers")
    parser.add_argument("--end-year", type=int, required=True, help="End year for papers")
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    papers_dir = repo_root / "pipeline" / "papers-data"
    company_dir = repo_root / "pipeline" / "company-data"

    fetch_and_clean_papers(args.start_year, args.end_year, papers_dir)
    fetch_crunchbase_companies(company_dir)

    subprocess.run([
        sys.executable, str(repo_root / "scripts" / "clean" / "clean_and_normalize.py"),
        "--papers-input", str(papers_dir / "cleaned_papers.jsonl"),
        "--papers-output", str(papers_dir / "papers_normalized.csv"),
        "--tech-input", str(repo_root / args.tech_csv),
        "--tech-output", str(repo_root / "pipeline" / "technologies_normalized.csv"),
    ], check=True)

    # Call enrich.py as a script, outputting JSON
    tech_norm_csv = str(repo_root / "pipeline" / "technologies_normalized.csv")
    tech_defs_json = str(repo_root / "pipeline" / "technologies_with_definitions.json")
    subprocess.run([
        "python", str(repo_root / "scripts" / "enrich" / "enrich.py"),
        "--input", tech_norm_csv,
        "--output", tech_defs_json
    ], check=True)
    
    subprocess.run([sys.executable, str(repo_root / "scripts" / "classify" / "classify_papers_v2.py"),],check=True)
    
    subprocess.run([sys.executable, str(repo_root / "scripts" / "linking" / "link_papers_to_technology.py"),
                    "--papers", str(papers_dir / "papers_classified.csv"),
                    "--techs", str(repo_root / "pipeline" / "technologies_normalized.csv"),
                    "--output", str(repo_root / "pipeline" / "neo4j-files" / "paper_technology_links.csv")],
                    check=True)
    
    subprocess.run([sys.executable, str(repo_root / "scripts" / "linking" / "full_linking_scripts.py"),])
    
    
    
    print(f"Pipeline finished. Data located at {repo_root / 'pipeline' / 'neo4j-files'}")


if __name__ == "__main__":
    main()
