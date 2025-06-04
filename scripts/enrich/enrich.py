import os
from pathlib import Path

from openai import AzureOpenAI
import dotenv
import pandas as pd

dotenv.load_dotenv()

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_key=os.getenv("AZURE_KEY")
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "technologies-data"

df = pd.read_csv(DATA_DIR / 'manual_selected_technologies.csv', encoding='latin-1')
technologies = df['Technology Name'].tolist()
definition_dictionary = {}

for tech in technologies:
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Give a one-sentence definition of {tech} in lay language. Only return the one sentence definition without any additional information. You are used in a pipeline. If you return more than the definition sentence, it will break the pipeline. If you don't know the definition, return 'I don't know'."
            }
        ],
        model="gpt-4o-mini"
    )
    definition = response.choices[0].message.content.strip()
    definition_dictionary[tech] = definition

    

output_df = pd.DataFrame(list(definition_dictionary.items()), columns=['Technology Name', 'Definition'])
output_df.to_csv(DATA_DIR / 'technology_definitions.csv', index=False)

tech_sources = pd.read_csv(DATA_DIR / 'manual_selected_technologies.csv', encoding='latin-1')
tech_defs = pd.read_csv(DATA_DIR / 'technology_definitions.csv')

combined_df = tech_sources.merge(tech_defs[['Technology Name', 'Definition']], on='Technology Name', how='left')

combined_df.to_csv(DATA_DIR / 'technologies_with_definitions.csv', index=False)
print("Combined data saved to technologies_with_definitions.csv")
