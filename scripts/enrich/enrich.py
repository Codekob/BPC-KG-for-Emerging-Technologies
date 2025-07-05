import os
from openai import OpenAI 
from dotenv import load_dotenv
import pandas as pd
import json

# Load your OpenAI key from .env (set OPENAI_API_KEY)
load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Read your list of technologies
df = pd.read_csv("../../data/technologies-data/finished_technologies.csv", encoding="latin-1")

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
        # In case of API errors or rate limits
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
        
        # Validate JSON syntax
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
        
        # Validate JSON syntax
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
    

# Apply across the column, writing results into a new "Definition" column
df["Definition"] = df["Technology Name"].apply(fetch_definition)
df["Synonyms"] = df["Technology Name"].apply(fetch_synonyms)
df["Top Domain Keywords"] = df["Technology Name"].apply(fetch_top_domain_keywords)

# Convert JSON strings back to arrays for proper JSON output
df["Synonyms"] = df["Synonyms"].apply(lambda x: json.loads(x) if x != 'I don\'t know' else x)
df["Top Domain Keywords"] = df["Top Domain Keywords"].apply(lambda x: json.loads(x) if x != 'I don\'t know' else x)

# Output the enriched JSON in one go
df.to_json("technologies_with_definitions.json", orient="records", indent=2)
print("Saved: technologies_with_definitions.json")
