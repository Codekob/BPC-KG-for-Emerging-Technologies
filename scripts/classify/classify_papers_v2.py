import os
import json
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

# CONFIGURATION: adjust thresholds and models here
EMB_MODEL    = 'paraphrase-mpnet-base-v2'   # embedding model
TH_EMB_HIGH  = 0.61     # high-confidence embedding match (lowered to boost recall)
TH_EMB_LOW   = 0.35     # low-confidence embedding gate
TH_TFIDF     = 0.35     # TF-IDF minimum
TH_FUZZY     = 0.60     # fuzzy match minimum
TITLE_BOOST  = True     # enable title boost
TITLE_FACTOR = 0.3     # additive boost (slightly reduced)
MIN_ADDITIONAL_SIGNALS = 1  # require at least 1 of {TF-IDF, fuzzy, keyword}

# Paths to data
PAPERS_PATH = os.path.expanduser('pipeline/papers-data/papers_normalized.csv')
DEFS_PATH   = os.path.expanduser('pipeline/technologies_with_definitions.json')

# Load data
def load_data():
    start = time.time()
    papers = pd.read_csv(PAPERS_PATH)
    with open(DEFS_PATH, 'r') as f:
        tech_defs = pd.DataFrame(json.load(f))
    print(f"Loaded {len(papers)} papers and {len(tech_defs)} tech defs in {time.time()-start:.1f}s")
    return papers, tech_defs

# Prepare technology descriptions + keyword sets
def build_tech_descriptions(tech_defs):
    start = time.time()
    recs = []
    for r in tech_defs.to_dict(orient='records'):
        desc = ' '.join([r['Technology Name'], r.get('Definition',''),
                         ' '.join(r.get('Synonyms',[])), ' '.join(r.get('Top Domain Keywords',[]))])
        kws = set([r['Technology Name'].lower()] +
                  [s.lower() for s in r.get('Synonyms',[])] +
                  [w.lower() for w in r.get('Top Domain Keywords',[])])
        recs.append({'tech_name': r['Technology Name'], 'description': desc, 'keywords': kws})
    df = pd.DataFrame(recs)
    print(f"Built {len(df)} tech descriptions in {time.time()-start:.1f}s")
    return df

# Combine relevant paper fields into one text column
def build_paper_corpus(papers):
    start = time.time()
    text_cols = ['title', 'abstract', 'title_norm', 'keywords_flat',
                 'topics_flat', 'subfields_flat', 'fields_flat']
    def concat(r): return ' '.join(str(r[c]) for c in text_cols if c in r and pd.notnull(r[c]))
    papers['combined'] = tqdm(papers.apply(concat, axis=1), desc='Building corpus', total=len(papers))
    print(f"Built corpus in {time.time()-start:.1f}s")
    return papers

# Core matching: sequential gating + multi-signal requirement
def match_technologies(papers, tech_descs):
    # 1) Embedding similarity
    t0 = time.time()
    model = SentenceTransformer(EMB_MODEL)
    tech_embs  = model.encode(tech_descs['description'].tolist(), normalize_embeddings=True)
    paper_embs = model.encode(papers['combined'].tolist(), normalize_embeddings=True)
    sim_emb = cosine_similarity(paper_embs, tech_embs)
    print(f"Embeddings computed in {time.time()-t0:.1f}s")

    # 2) TF-IDF similarity
    t1 = time.time()
    vect = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=5000)
    corpus = tech_descs['description'].tolist() + papers['combined'].tolist()
    tfidf = vect.fit_transform(corpus)
    sim_tfidf = cosine_similarity(tfidf[len(tech_descs):], tfidf[:len(tech_descs)])
    print(f"TF-IDF sim in {time.time()-t1:.1f}s")

    # 3) Prepare keyword sets
    t2 = time.time()
    paper_kw_sets = papers['keywords_flat'].fillna('').str.lower().str.split('[,;]') \
                        .apply(lambda ws: set(w.strip() for w in ws if w.strip()))
    print(f"Keyword sets in {time.time()-t2:.1f}s")

    # 4) Lowercase tech names for title boost
    tech_names_lower = tech_descs['tech_name'].str.lower().tolist()

    # 5) Loop papers & assign
    t3 = time.time()
    preds = []
    for i, row in enumerate(tqdm(papers.itertuples(index=False), desc='Classifying', total=len(papers))):
        text = row.combined
        title = str(getattr(row, 'title_norm', '') or '')
        emb_scores = sim_emb[i].copy()
        if TITLE_BOOST and title:
            tl = title.lower()
            for j, tech_l in enumerate(tech_names_lower):
                if tech_l in tl:
                    emb_scores[j] = min(1.0, emb_scores[j] + TITLE_FACTOR)

        idxs = np.where(emb_scores >= TH_EMB_HIGH)[0]
        if len(idxs) == 0:
            weak = np.where(emb_scores >= TH_EMB_LOW)[0]
            for j in weak:
                count = 0
                if sim_tfidf[i,j] >= TH_TFIDF: count += 1
                if fuzz.token_set_ratio(tech_descs.loc[j,'description'], text)/100. >= TH_FUZZY: count += 1
                if paper_kw_sets[i] & tech_descs.loc[j,'keywords']: count += 1
                if count >= MIN_ADDITIONAL_SIGNALS:
                    idxs = np.append(idxs, j)
        preds.append('; '.join(tech_descs.loc[sorted(set(idxs)),'tech_name']))
    print(f"Classified in {time.time()-t3:.1f}s")

    papers['predicted_technology'] = preds
    return papers

# Save to disk
def save_classified(papers, out_path='pipeline/papers-data/papers_classified.csv'):
    t4 = time.time()
    papers.to_csv(out_path, index=False)
    print(f"Saved file in {time.time()-t4:.1f}s at {out_path}")

# Main
def main():
    t_start = time.time()
    papers, tech_defs = load_data()
    tech_descs = build_tech_descriptions(tech_defs)
    papers = build_paper_corpus(papers)
    papers = match_technologies(papers, tech_descs)
    save_classified(papers)
    print(f"Total runtime {time.time()-t_start:.1f}s")

if __name__=='__main__':
    main()
