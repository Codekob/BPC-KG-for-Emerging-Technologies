#!/usr/bin/env python3
"""
build_entity_nodes.py

Produces three CSVs of node metadata for Neo4j:

  1. technologies_nodes.csv
     • tech_name
     • description

  2. papers_nodes.csv (JSON Lines, OpenAlex schema)
     • paper_id         (rec['id'])
     • link             (rec['doi'])
     • title            (rec['title'])
     • authors          (semicolon-separated rec['authorships'][*]['display_name'])
     • pub_date         (rec['publication_date'])
     • institution      (semicolon-separated rec['institutions'][*]['display_name'])

  3. companies_nodes.csv (pure JSON, Crunchbase schema)
     • company_id
     • company_name
     • founding_date
     • location
     • funding_total_usd
     • funding_stage
"""

import json
import pandas as pd
import pathlib

# ─── PATHS ─────────────────────────────────────────────────────────────────────
BASE         = pathlib.Path('../../data')

TECH_JSON    = BASE / 'technologies-data' / 'technologies_with_definitions.json'
PAPERS_JSONL = BASE / 'papers-data'       / 'cleaned_papers.jsonl'
COMP_JSON    = BASE / 'company-data'      / 'crunchbase_fetching_raw.json'

OUT_DIR      = BASE / 'neo4j-nodes'
OUT_DIR.mkdir(exist_ok=True)

TECH_OUT     = OUT_DIR / 'technologies_nodes.csv'
PAPER_OUT    = OUT_DIR / 'papers_nodes.csv'
COMP_OUT     = OUT_DIR / 'companies_nodes.csv'


# ─── 1) TECHNOLOGIES ───────────────────────────────────────────────────────────
def build_technologies(nodes_out):
    with open(TECH_JSON, 'r', encoding='utf-8') as f:
        tech_defs = json.load(f)

    rows = []
    for rec in tech_defs:
        name       = (rec.get('Technology Name') or '').strip()
        parts      = [name]
        definition = (rec.get('Definition') or '').strip()
        if definition:
            parts.append(definition)
        for syn in rec.get('Synonyms') or []:
            s = (syn or '').strip()
            if s:
                parts.append(s)
        for kw in rec.get('Top Domain Keywords') or []:
            k = (kw or '').strip()
            if k:
                parts.append(k)
        description = ' '.join(parts)

        rows.append({'tech_name': name,
                     'description': description})

    pd.DataFrame(rows).to_csv(nodes_out, index=False, encoding='utf-8')
    print(f"Wrote {len(rows)} technology nodes to {nodes_out}")


# ─── 2) PAPERS (JSONL, OpenAlex‐style) ─────────────────────────────────────────
def build_papers(nodes_out):
    rows = []
    with open(PAPERS_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            # id and link
            paper_id = (rec.get('id') or '').strip()
            link     = (rec.get('doi') or '').strip()

            # title
            title = (rec.get('title') or '').strip()

            # authors
            auths = rec.get('authorships') or []
            authors = '; '.join(
                a.get('display_name', '').strip()
                for a in auths
                if isinstance(a, dict) and a.get('display_name')
            )

            # publication date
            pub_date = (rec.get('publication_date') or '').strip()

            # institutions
            insts = rec.get('institutions') or []
            institution = '; '.join(
                i.get('display_name', '').strip()
                for i in insts
                if isinstance(i, dict) and i.get('display_name')
            )

            rows.append({
                'paper_id':   paper_id,
                'link':       link,
                'title':      title,
                'authors':    authors,
                'pub_date':   pub_date,
                'institution': institution
            })

    pd.DataFrame(rows).to_csv(nodes_out, index=False, encoding='utf-8')
    print(f"Wrote {len(rows)} paper nodes to {nodes_out}")


# ─── 3) COMPANIES (pure JSON, Crunchbase) ──────────────────────────────────────
def build_companies(nodes_out):
    with open(COMP_JSON, 'r', encoding='utf-8') as f:
        entries = json.load(f)

    rows = []
    for e in entries:
        props = e.get('properties') or {}

        # identifier
        ident = props.get('identifier') or {}
        cid   = (ident.get('permalink') or '').strip()
        name  = (ident.get('value')     or '').strip()

        # founding_date (string or dict)
        fr = props.get('founded_on') or ''
        if isinstance(fr, dict):
            founded = (fr.get('value') or '').strip()
        else:
            founded = str(fr).strip()

        # location: city → region → country
        locs   = props.get('location_identifiers') or []
        city   = next((i.get('value') for i in locs if i.get('location_type')=='city'), '')
        region = next((i.get('value') for i in locs if i.get('location_type')=='region'), '')
        country= next((i.get('value') for i in locs if i.get('location_type')=='country'), '')
        location = city or region or country or ''

        # funding_total.value_usd
        ft = props.get('funding_total') or {}
        funding_usd = ft.get('value_usd') or ''

        # funding_stage
        stage = (props.get('funding_stage') or '').strip()

        rows.append({
            'company_id':        cid,
            'company_name':      name,
            'founding_date':     founded,
            'location':          location,
            'funding_total_usd': funding_usd,
            'funding_stage':     stage
        })

    pd.DataFrame(rows).to_csv(nodes_out, index=False, encoding='utf-8')
    print(f"Wrote {len(rows)} company nodes to {nodes_out}")


if __name__ == '__main__':
    build_technologies(TECH_OUT)
    build_papers(PAPER_OUT)
    build_companies(COMP_OUT)
    print("All three node CSVs written to:", OUT_DIR)
