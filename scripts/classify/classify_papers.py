#!/usr/bin/env python3
"""
scripts/classify/classify_papers_by_embedding.py

Classify each paper into the best-matching technology using hybrid matching:
 - Field-exact for full tech phrase
 - Semantic embedding primary matching
 - Token-level fallback matching
 - Fuzzy backup if needed

Inputs:
 - ../../data/papers-data/papers_normalized.csv
 - ../../data/technologies-data/technologies_normalized.csv
 - ../../data/technologies-data/technology_definitions.csv

Outputs:
 - ../../data/papers-data/papers_classified.csv  # includes 'predicted_technology', 'match_type', 'similarity'

Dependencies:
 - pandas, numpy, sentence-transformers, scikit-learn, rapidfuzz
"""
import pandas as pd
import numpy as np
import string
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
import argparse

# --- Text normalization ---
def normalize_text(s: str) -> str:
    return (
        s.lower()
         .translate(str.maketrans('', '', string.punctuation))
         .strip()
         .replace('\n',' ')
    )

# --- Load inputs ---
def load_inputs(
    papers_path: str = "../../data/papers-data/papers_normalized.csv",
    techs_path: str = "../../data/technologies-data/technologies_normalized.csv",
    defs_path: str = "../../data/technologies-data/technologies_with_definitions.csv"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    papers = pd.read_csv(papers_path)
    techs = pd.read_csv(techs_path)
    defs = pd.read_csv(defs_path)
    techs = techs.merge(defs, on='Technology Name', how='left')
    techs['tech_norm'] = techs['Technology Name'].map(normalize_text)
    return papers, techs

# --- Compute embeddings ---
def compute_embeddings(
    papers: pd.DataFrame, techs: pd.DataFrame,
    model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'
) -> tuple[np.ndarray, np.ndarray]:
    model = SentenceTransformer(model_name)
    paper_texts = (
        papers.get('keywords_flat','').fillna('') + ' ' +
        papers.get('topics_flat','').fillna('') + ' ' +
        papers.get('subfields_flat','').fillna('') + ' ' +
        papers.get('fields_flat','').fillna('') + ' ' +
        papers.get('title','').fillna('') + ' ' +
        papers.get('abstract','').fillna('')
    ).tolist()
    tech_texts = (
        techs['Technology Name'].fillna('') + '. ' + techs['Definition'].fillna('')
    ).tolist()
    paper_embeds = model.encode(paper_texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)
    tech_embeds  = model.encode(tech_texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)
    return paper_embeds, tech_embeds

# --- Classification ---
def classify_papers(
    papers: pd.DataFrame,
    techs: pd.DataFrame,
    paper_embeds: np.ndarray,
    tech_embeds: np.ndarray,
    emb_threshold: float = 0.3,
    fuzzy_threshold: int = 70
) -> pd.DataFrame:
    sims = cosine_similarity(paper_embeds, tech_embeds)
    results = []
    fields = ['keywords_flat','topics_flat','subfields_flat','fields_flat','title','abstract']

    for i, row in papers.iterrows():
        predicted = ''
        match_type = ''
        similarity = 0.0

        # 1) Field-exact match for full tech phrase
        for tech_norm, tech_name in zip(techs['tech_norm'], techs['Technology Name']):
            for fld in fields:
                if tech_norm and tech_norm in normalize_text(str(row.get(fld,''))):
                    predicted = tech_name
                    match_type = 'field_exact'
                    similarity = 1.0
                    break
            if predicted:
                break

        # 2) Embedding-based match (primary)
        if not predicted:
            sim_scores = sims[i]
            best_idx = int(np.argmax(sim_scores))
            best_score = float(sim_scores[best_idx])
            if best_score >= emb_threshold:
                predicted = techs.loc[best_idx, 'Technology Name']
                match_type = 'embedding'
                similarity = best_score

        # 3) Token-level fallback
        if not predicted:
            for tech_norm, tech_name in zip(techs['tech_norm'], techs['Technology Name']):
                tokens = [tok for tok in tech_norm.split() if len(tok) > 4]
                for tok in tokens:
                    for fld in fields:
                        if tok in normalize_text(str(row.get(fld,''))):
                            predicted = tech_name
                            match_type = 'token_match'
                            similarity = 0.6
                            break
                    if predicted:
                        break
                if predicted:
                    break

        # 4) Fuzzy fallback
        if not predicted:
            combined = normalize_text(' '.join(str(row.get(f,'')) for f in fields))
            bestMatch, score = max(
                [(tn, fuzz.token_sort_ratio(normalize_text(tn), combined)) for tn in techs['Technology Name']],
                key=lambda x: x[1]
            )
            if score >= fuzzy_threshold:
                predicted = bestMatch
                match_type = 'fuzzy_fallback'
                similarity = score/100.0

        results.append({
            'predicted_technology': predicted,
            'match_type': match_type,
            'similarity': round(similarity, 3)
        })

    return pd.concat([papers.reset_index(drop=True), pd.DataFrame(results)], axis=1)

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Classify papers into technologies")
    parser.add_argument("--papers", default="../../data/papers-data/papers_normalized.csv", help="Input papers CSV")
    parser.add_argument("--techs", default="../../data/technologies-data/technologies_normalized.csv", help="Input technologies CSV")
    parser.add_argument("--defs", default="../../data/technologies-data/technologies_with_definitions.csv", help="Input technology definitions CSV")
    parser.add_argument("--output", default="../../data/papers-data/papers_classified.csv", help="Output classified papers CSV")
    args = parser.parse_args()

    papers, techs = load_inputs(args.papers, args.techs, args.defs)
    p_emb, t_emb = compute_embeddings(papers, techs)
    classified = classify_papers(papers, techs, p_emb, t_emb)
    classified.to_csv(args.output, index=False)
    print(f"Wrote {len(classified)} classified papers to {args.output}")

if __name__=='__main__':
    main()
