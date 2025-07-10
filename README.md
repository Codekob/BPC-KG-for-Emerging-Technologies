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
python pipeline/run_pipeline.py \
  --tech-csv pipeline/technlist.csv \
  --start-year 2019 --end-year 2024
```

The resulting Neo4j import file will be written to
`pipeline/neo4j-files`.

