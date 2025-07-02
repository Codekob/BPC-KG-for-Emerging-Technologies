import os
from openai import OpenAI 
from dotenv import load_dotenv
import pandas as pd

# Load your OpenAI key from .env (set OPENAI_API_KEY)
load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Read your list of technologies
df = pd.read_csv("../../data/technologies-data/finished_technologies.csv", encoding="latin-1")

def fetch_definition(tech: str) -> str:
    """Return a one-sentence lay definition, or 'I don't know'."""

    prompt = f"Give a one-sentence definition of {tech} in lay language. Only return the one sentence definition without any additional information. You are used in a pipeline. If you return more than the definition sentence, it will break the pipeline. If you don't know the definition, return 'I don't know'."
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

# Apply across the column, writing results into a new "Definition" column
df["Definition"] = df["Technology Name"].apply(fetch_definition)

# Output the enriched CSV in one go
df.to_csv("technologies_with_definitions.csv", index=False)
print("Saved: technologies_with_definitions.csv")
