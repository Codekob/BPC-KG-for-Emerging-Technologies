import os
from openai import OpenAI 
from dotenv import load_dotenv
import pandas as pd
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load your OpenAI key from .env (set OPENAI_API_KEY)
load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

def fetch_definition(tech: str) -> str:
    """Return a one-sentence lay definition, or 'I don't know'."""
    prompt = f"Give a 3-4 sentence definition of the Technology: {tech} in lay language. Only return the one sentence definition without any additional information. You are used in a pipeline. If you return more than the definition sentence, it will break the pipeline. If you don't know the definition, return 'I don't know'."
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
        )
        return response.output_text
    except Exception as e:
        print(f"Error fetching definition for {tech}: {e}")
        return
    
def fetch_synonyms(tech: str) -> str:
    """Return a JSON array of synonyms, or 'I don't know'."""
    prompt = f"Give a JSON array of the top 3 synonyms for the emerging technology: {tech}. Return only a valid JSON array like ['synonym1', 'synonym2', 'synonym3']. If you don't know any, return 'I don't know'. Only return the JSON array without any additional information. You are used in a pipeline."
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
        )
        result = response.output_text.strip()
        if result != 'I don\'t know':
            try:
                json.loads(result)
                return result
            except json.JSONDecodeError:
                print(f"Invalid JSON for {tech} synonyms: {result}")
                return 'I don\'t know'
        return result
    except Exception as e:
        print(f"Error fetching synonyms for {tech}: {e}")
        return
    
def fetch_top_domain_keywords(tech: str) -> str:
    """Return a JSON array of top domain keywords, or 'I don't know'."""
    prompt = f"Give a JSON array of the top 5 domain keywords for the emerging technology: {tech}. Return only a valid JSON array like ['keyword1', 'keyword2', 'keyword3', 'keyword4', 'keyword5']. If you don't know any, return 'I don't know'. Only return the JSON array without any additional information. You are used in a pipeline."
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
        )
        result = response.output_text.strip()
        if result != 'I don\'t know':
            try:
                json.loads(result)
                return result
            except json.JSONDecodeError:
                print(f"Invalid JSON for {tech} keywords: {result}")
                return 'I don\'t know'
        return result
    except Exception as e:
        print(f"Error fetching top domain keywords for {tech}: {e}")
        return

def enrich_technologies(input_csv, output_json):
    df = pd.read_csv(input_csv, encoding="latin-1")
    if "Technology Name" not in df.columns:
        df.columns = ["Technology Name"]

    tech_names = df["Technology Name"].tolist()
    results = [None] * len(tech_names)

    def enrich_one(idx_tech):
        idx, tech = idx_tech
        definition = fetch_definition(tech)
        synonyms = fetch_synonyms(tech)
        keywords = fetch_top_domain_keywords(tech)
        return idx, definition, synonyms, keywords

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(enrich_one, (i, tech)): i for i, tech in enumerate(tech_names)}
        for future in as_completed(futures):
            idx, definition, synonyms, keywords = future.result()
            results[idx] = (definition, synonyms, keywords)

    df["Definition"] = [r[0] for r in results]
    df["Synonyms"] = [r[1] for r in results]
    df["Top Domain Keywords"] = [r[2] for r in results]
    df["Synonyms"] = df["Synonyms"].apply(lambda x: json.loads(x) if x != 'I don\'t know' else x)
    df["Top Domain Keywords"] = df["Top Domain Keywords"].apply(lambda x: json.loads(x) if x != 'I don\'t know' else x)
    df.to_json(output_json, orient="records", indent=2)
    print(f"Saved: {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich technologies with definitions, synonyms, and keywords.")
    parser.add_argument("--input", type=str, default="../../data/technologies-data/finished_technologies.csv", help="Input CSV file with technologies")
    parser.add_argument("--output", type=str, default="technologies_with_definitions.json", help="Output JSON file")
    args = parser.parse_args()
    print("Enriching technologies with definitions, synonyms, and keywords...")
    enrich_technologies(args.input, args.output)