#!/usr/bin/env python3
"""
link_comp_to_technology.py

1) Extract Crunchbase company data (id, name, description, categories)
   into a CSV.
2) Match companies to technologies via:
     - Direct category↔keyword overlap (≥70%, with normalization & fuzzy)
       * also direct-match against the technology’s own name
     - Semantic embeddings (Sentence-Transformers)
     - TF-IDF cosine similarity
     - RapidFuzz fuzzy matching
   with configurable thresholds.
3) Output:
     - companies_classified.csv   (with predicted_technology column)
     - company_technology_links.csv  (flat company_id→technology_name)
"""

import os
import json
import time
import ast
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
RAW_JSON         = '../../data/company-data/crunchbase_fetching_raw.json'
EMBED_INPUT_CSV  = '../../data/company-data/companies_embedding_input.csv'
TECHS_JSON       = '../../data/technologies-data/technologies_with_definitions.json'
OUT_CLASSIFIED   = '../../data/company-data/companies_classified.csv'
OUT_LINKS        = '../../data/linking-data/company_technology_links.csv'

# Direct category matching thresholds
CATEGORY_THRESH   = 0.40   # require ≥70% of tech keywords matched
FUZZY_CAT_THRESH  = 75     # rapidfuzz.token_sort_ratio threshold

# Embedding & auxiliary thresholds
EMB_MODEL        = 'paraphrase-mpnet-base-v2'
TH_EMB_HIGH      = 0.49
TH_EMB_LOW       = 0.35
TH_TFIDF         = 0.28
TH_FUZZY         = 0.45
MIN_ADDITIONAL   = 1
NAME_BOOST       = 0.50

# ─── STEP 1: EXTRACT COMPANIES FROM RAW JSON ────────────────────────────────────
def extract_and_dump_companies(json_path, out_csv):
    t0 = time.time()
    with open(json_path, 'r', encoding='utf-8') as f:
        entries = json.load(f)
    rows = []
    for e in entries:
        props = e.get('properties', {})
        cid  = props.get('identifier', {}).get('permalink', '')
        name = props.get('identifier', {}).get('value', '')
        desc = props.get('description', '')
        cats = [c.get('permalink','') for c in props.get('categories',[]) if c.get('permalink')]
        rows.append({
            'company_id': cid,
            'company_name': name,
            'description': desc,
            'categories': cats
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding='utf-8')
    print(f"Wrote {len(df)} records to {out_csv} in {time.time()-t0:.2f}s")
    return df

# ─── STEP 2: LOAD COMPANY & TECHNOLOGY DATA ────────────────────────────────────
def load_data(comp_csv, techs_json):
    t0 = time.time()
    df = pd.read_csv(comp_csv, encoding='utf-8')
    df['categories'] = df['categories'].apply(ast.literal_eval)
    with open(techs_json, 'r', encoding='utf-8') as f:
        tech_defs = pd.DataFrame(json.load(f))
    print(f"Loaded {len(df)} companies and {len(tech_defs)} tech definitions in {time.time()-t0:.2f}s")
    return df, tech_defs

# ─── STEP 3: BUILD TECH DESCRIPTIONS + KEYWORD SETS ─────────────────────────────
def build_tech_descriptions(defs):
    recs = []
    for r in defs.to_dict('records'):
        name = r['Technology Name']
        text = ' '.join([
            name,
            r.get('Definition',''),
            *r.get('Synonyms',[]),
            *r.get('Top Domain Keywords',[])
        ])
        kw_set = set([name.lower()] +
                     [s.lower() for s in r.get('Synonyms',[])] +
                     [k.lower() for k in r.get('Top Domain Keywords',[])])
        recs.append({'tech_name': name,
                     'description': text,
                     'keywords': kw_set})
    tech_df = pd.DataFrame(recs)
    print(f"Prepared {len(tech_df)} tech descriptions")
    return tech_df

# ─── STEP 4: BUILD COMPANY CORPUS (INCLUDING CATEGORIES) ────────────────────────
def build_company_corpus(comps):
    t0 = time.time()
    def concat(r):
        parts = [
            str(r.company_name),
            str(r.description) if pd.notnull(r.description) else ''
        ]
        if r.categories:
            parts.append(' '.join(r.categories))
        return ' '.join(parts)

    comps['combined'] = tqdm(
        comps.apply(concat, axis=1),
        desc='Building company corpus',
        total=len(comps)
    )
    print(f"Built company corpus in {time.time()-t0:.2f}s")
    return comps

# ─── STEP 5: MATCHING ────────────────────────────────────────────────────────────
def match_companies(comps, tech_df):
    # normalize helper
    def normalize(s):
        return re.sub(r'[-_]+', ' ', s.lower()).strip()

    # tech keywords + normalized tech names
    tech_kw_sets_norm = []
    tech_name_norms   = []
    for kw_set, tech_name in zip(tech_df['keywords'], tech_df['tech_name']):
        norm_set = set()
        for kw in kw_set:
            norm_set.add(normalize(kw))
        tech_kw_sets_norm.append(norm_set)
        tech_name_norms.append(normalize(tech_name))

    # normalize company categories
    comp_cat_lists_norm = [
        [normalize(c) for c in cats]
        for cats in comps['categories']
    ]

    # direct category↔keyword & categor tech-name linking
    direct_preds = [set() for _ in range(len(comps))]
    for i, cat_list in enumerate(comp_cat_lists_norm):
        for j, (kw_set, tech_name_norm) in enumerate(zip(tech_kw_sets_norm, tech_name_norms)):
            # first check tech name
            if any(cat == tech_name_norm or fuzz.token_sort_ratio(cat, tech_name_norm) >= FUZZY_CAT_THRESH
                   for cat in cat_list):
                direct_preds[i].add(tech_df.at[j, 'tech_name'])
                continue
            # otherwise count matched keywords
            if not kw_set:
                continue
            matched = 0
            for kw in kw_set:
                if any(cat == kw or fuzz.token_sort_ratio(cat, kw) >= FUZZY_CAT_THRESH
                       for cat in cat_list):
                    matched += 1
            if (matched / len(kw_set)) >= CATEGORY_THRESH:
                direct_preds[i].add(tech_df.at[j, 'tech_name'])

    # embedding similarity
    t0 = time.time()
    model     = SentenceTransformer(EMB_MODEL)
    tech_embs = model.encode(tech_df.description.tolist(), normalize_embeddings=True)
    comp_embs = model.encode(comps.combined.tolist(),     normalize_embeddings=True)
    sim_emb   = cosine_similarity(comp_embs, tech_embs)
    print(f"Computed embeddings in {time.time()-t0:.2f}s")

    # TF-IDF similarity
    t1 = time.time()
    vect       = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=5000)
    corpus     = tech_df.description.tolist() + comps.combined.tolist()
    tfidf      = vect.fit_transform(corpus)
    tech_tfidf = tfidf[:len(tech_df)]
    comp_tfidf = tfidf[len(tech_df):]
    sim_tfidf  = cosine_similarity(comp_tfidf, tech_tfidf)
    print(f"TF-IDF sim computed in {time.time()-t1:.2f}s")

    tech_names_lower = tech_df.tech_name.str.lower().tolist()
    preds = []
    t2 = time.time()
    for i, row in enumerate(tqdm(comps.itertuples(index=False),
                                desc='Matching companies',
                                total=len(comps))):
        linked = set(direct_preds[i])

        # name boosting in embeddings
        emb_scores = sim_emb[i].copy()
        name_low   = row.company_name.lower() if row.company_name else ''
        for j, tech_l in enumerate(tech_names_lower):
            if tech_l in name_low:
                emb_scores[j] = min(1.0, emb_scores[j] + NAME_BOOST)

        # high-confidence embeddings
        high_idxs = np.where(emb_scores >= TH_EMB_HIGH)[0]
        for j in high_idxs:
            linked.add(tech_df.at[j, 'tech_name'])

        # weaker embeddings + auxiliary signals
        if len(high_idxs) == 0:
            weak_idxs = np.where(emb_scores >= TH_EMB_LOW)[0]
            for j in weak_idxs:
                count = 0
                if sim_tfidf[i, j] >= TH_TFIDF:
                    count += 1
                if fuzz.token_set_ratio(tech_df.loc[j,'description'], row.combined) / 100. >= TH_FUZZY:
                    count += 1
                if count >= MIN_ADDITIONAL:
                    linked.add(tech_df.at[j, 'tech_name'])

        preds.append('; '.join(sorted(linked)))

    print(f"Finished matching in {time.time()-t2:.2f}s")
    comps['predicted_technology'] = preds
    return comps

# ─── STEP 6: SAVE OUTPUTS ───────────────────────────────────────────────────────
def save_results(comps, out_classified, out_links):
    t0 = time.time()
    comps.to_csv(out_classified, index=False)
    print(f"Saved classified companies to {out_classified} in {time.time()-t0:.2f}s")

    links = []
    for _, r in comps.iterrows():
        for tech in r.predicted_technology.split(';'):
            tech = tech.strip()
            if tech:
                links.append({'company_id': r.company_id,
                              'technology_name': tech})
    links_df = pd.DataFrame(links)
    links_df.to_csv(out_links, index=False)
    print(f"Saved links to {out_links} in {time.time()-t0:.2f}s")

def main():
    extract_and_dump_companies(RAW_JSON, EMBED_INPUT_CSV)
    comps, tech_defs = load_data(EMBED_INPUT_CSV, TECHS_JSON)
    tech_df         = build_tech_descriptions(tech_defs)
    comps           = build_company_corpus(comps)
    comps           = match_companies(comps, tech_df)
    save_results(comps, OUT_CLASSIFIED, OUT_LINKS)
    print("All done.")

if __name__ == '__main__':
    main()
