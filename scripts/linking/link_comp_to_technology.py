#!/usr/bin/env python3
"""
link_companies_to_technologies.py

Match Crunchbase companies to technologies:

  - Semantic embeddings (Sentence-Transformers)
  - TF-IDF cosine similarity
  - RapidFuzz fuzzy matching
  - (No category signal—none in your CSV)

Outputs:
  - companies_classified.csv   (adds predicted_technology)
  - company_technology_links.csv  (flat company_id→technology_name)
"""

import os
import json
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
EMB_MODEL      = 'paraphrase-mpnet-base-v2'
TH_EMB_HIGH    = 0.49   # high-confidence embedding
TH_EMB_LOW     = 0.35    # fallback embedding gate
TH_TFIDF       = 0.28    # TF-IDF gate
TH_FUZZY       = 0.45    # fuzzy gate
MIN_ADDITIONAL = 1       # auxiliary signals needed for weak embeddings
NAME_BOOST     = 0.50    # add if tech name in company_name

# ─── PATHS ─────────────────────────────────────────────────────────────────────
COMP_CSV        = os.path.expanduser(
    '../../data/company-data/crunchbase_fetch_linking.csv')
TECHS_JSON      = os.path.expanduser(
    '../../data/technologies-data/technologies_with_definitions.json')
OUT_CLASSIFIED  = os.path.expanduser(
    '../../data/company-data/companies_classified.csv')
OUT_LINKS       = os.path.expanduser(
    '../../data/linking-data/company_technology_links.csv')

# ─── LOAD DATA ─────────────────────────────────────────────────────────────────
def load_data():
    t0 = time.time()
    df = pd.read_csv(COMP_CSV)
    # rename for convenience
    df = df.rename(columns={
      'identifier.value': 'company_name',
      'identifier.permalink': 'company_id'
    })
    with open(TECHS_JSON) as f:
        tech_defs = pd.DataFrame(json.load(f))
    print(f"Loaded {len(df)} companies and "
          f"{len(tech_defs)} tech definitions in {time.time()-t0:.2f}s")
    return df, tech_defs

# ─── BUILD TECH DESCRIPTIONS ───────────────────────────────────────────────────
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

# ─── BUILD COMPANY CORPUS ──────────────────────────────────────────────────────
def build_company_corpus(comps):
    t0 = time.time()
    def concat(r):
        return ' '.join([
            str(r.company_name),
            str(r.description) if pd.notnull(r.description) else ''
        ])
    comps['combined'] = tqdm(
        comps.apply(concat, axis=1),
        desc='Building company corpus',
        total=len(comps)
    )
    print(f"Built company corpus in {time.time()-t0:.2f}s")
    return comps

# ─── MATCHING ───────────────────────────────────────────────────────────────────
def match_companies(comps, tech_df):
    # Embedding similarity
    t0 = time.time()
    model = SentenceTransformer(EMB_MODEL)
    tech_embs = model.encode(
        tech_df.description.tolist(),
        normalize_embeddings=True
    )
    comp_embs = model.encode(
        comps.combined.tolist(),
        normalize_embeddings=True
    )
    sim_emb = cosine_similarity(comp_embs, tech_embs)
    print(f"Computed embeddings in {time.time()-t0:.2f}s")

    # TF-IDF similarity
    t1 = time.time()
    vect = TfidfVectorizer(
        stop_words='english', ngram_range=(1,2), max_features=5000
    )
    corpus = tech_df.description.tolist() + comps.combined.tolist()
    tfidf = vect.fit_transform(corpus)
    tech_tfidf = tfidf[:len(tech_df)]
    comp_tfidf = tfidf[len(tech_df):]
    sim_tfidf = cosine_similarity(comp_tfidf, tech_tfidf)
    print(f"TF-IDF sim computed in {time.time()-t1:.2f}s")

    # No category sets: dummy empty sets
    comp_kw_sets = [set() for _ in range(len(comps))]

    tech_names_lower = tech_df.tech_name.str.lower().tolist()

    preds = []
    t2 = time.time()
    for i, row in enumerate(tqdm(
        comps.itertuples(index=False),
        desc='Matching companies',
        total=len(comps)
    )):
        emb_scores = sim_emb[i].copy()
        text = row.combined
        name_low = row.company_name.lower() if row.company_name else ''
        # boost on name match
        for j, tech_l in enumerate(tech_names_lower):
            if tech_l in name_low:
                emb_scores[j] = min(1.0, emb_scores[j] + NAME_BOOST)

        # high-confidence
        idxs = np.where(emb_scores >= TH_EMB_HIGH)[0]
        if len(idxs) == 0:
            # weaker gate
            weak = np.where(emb_scores >= TH_EMB_LOW)[0]
            for j in weak:
                count = 0
                if sim_tfidf[i,j] >= TH_TFIDF: count += 1
                if fuzz.token_set_ratio(
                    tech_df.loc[j,'description'], text
                )/100. >= TH_FUZZY: count += 1
                # no keyword signal
                if count >= MIN_ADDITIONAL:
                    idxs = np.append(idxs, j)

        names = tech_df.loc[np.unique(idxs), 'tech_name'].tolist()
        preds.append('; '.join(names))

    print(f"Finished matching in {time.time()-t2:.2f}s")
    comps['predicted_technology'] = preds
    return comps

# ─── SAVE ───────────────────────────────────────────────────────────────────────
def save_results(comps):
    t0 = time.time()
    comps.to_csv(OUT_CLASSIFIED, index=False)
    print(f"Saved classified companies to {OUT_CLASSIFIED} in {time.time()-t0:.2f}s")

    links = []
    for _, r in comps.iterrows():
        cid = r.company_id
        for tech in r.predicted_technology.split(';'):
            tech = tech.strip()
            if tech:
                links.append({'company_id': cid, 'technology_name': tech})
    links_df = pd.DataFrame(links)
    t1 = time.time()
    links_df.to_csv(OUT_LINKS, index=False)
    print(f"Saved links to {OUT_LINKS} in {time.time()-t1:.2f}s")

def main():
    start = time.time()
    comps, tech_defs = load_data()
    tech_df = build_tech_descriptions(tech_defs)
    comps   = build_company_corpus(comps)
    comps   = match_companies(comps, tech_df)
    save_results(comps)
    print(f"Total runtime: {time.time()-start:.2f}s")

if __name__ == '__main__':
    main()
