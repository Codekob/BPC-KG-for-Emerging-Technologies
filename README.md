## Setup

1. Copy the example environment file and fill in your details:
   ```bash
   cp .env.example .env
   ```

2. Edit the .env file to set your email address, this is because the OpenAlex API requires an email for priority access

3. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```


**Normalize** papers and technologies.
   This step produces `papers_normalized.csv` and `technologies_normalized.csv`
   under `data/`.
   ```bash
   python scripts/clean/clean_and_normalize.py
   ```
2. **Classify** the normalized papers to assign a `predicted_technology`.
   Running the classification script creates `papers_classified.csv`.
   ```bash
   python scripts/enrich/classify_papers.py
   ```
3. **Link** papers to technologies using the classified file.
   ```bash
   python scripts/link/link_papers_to_technologies.py
   ```
   This outputs `paper_technology_links.csv` under `data/papers-data/` for
   loading into Neo4j.