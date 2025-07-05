## Setup

1. Copy the example environment file and fill in your details:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file to set your email address for OpenAlex as well as your
   `OPENAI_API_KEY` and `CRUNCHBASE_API_KEY`.

3. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```


## Running the pipeline

Run the entire process with one command. Provide the CSV of technologies and the
year range for the papers you want to fetch:

```bash
python scripts/run_pipeline.py \
  --tech-csv data/technologies-data/manual_selected_technologies.csv \
  --start-year 2019 --end-year 2024
```

The resulting Neo4j import file will be written to
`data/papers-data/paper_technology_links.csv`.

### Manual steps

If you prefer to run each stage separately:

1. **Normalize** papers and technologies.
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