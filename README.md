# Knowledge Graph for Emerging Technologies
This project was developed as part of the Bachelor Practical Course on Data Engineering at TUM. It provides a modular Python pipeline for constructing a knowledge graph that links emerging technologies, research papers, and startups.

*For the guide on how to run the pipeline, scroll down to* **Running the Pipeline** 

## Project Overview
The pipeline curates a list of emerging technologies and automatically collects:

- High-impact papers via the OpenAlex API (top 1,000 cited papers per year, 2019–2024)
- Leading startups via the Crunchbase API (top 800 European startups founded after 2020)

Entities are linked using:
- Deterministic name matching
- Transformer-based sentence embeddings (using Sentence-Transformers)
- TF-IDF and fuzzy-matching as fallback methods

### Outputs
The pipeline generates six CSV files:
- 3 Node files: Technologies, Papers, Companies
- 3 Edge files: Paper–Technology, Company–Technology, Company–Paper

These files are ready for direct bulk import into Neo4j, allowing immediate querying via Cypher.

How to Use

## Setup

1. Copy the example environment file and fill in your details:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file to set your email address for OpenAlex as well as your
   `OPENAI_API_KEY` and `CRUNCHBASE_API_KEY`.

   - **If you do not have a Crunchbase API key**, you can still run the pipeline! Set `CRUNCHBASE_API_KEY=NO_KEY` in your `.env` file. In this case, the pipeline will use a static snapshot of company data (as of 1 July 2025) located at `pipeline/company-data/crunchbase_fetching_raw.json` instead of fetching fresh data from the Crunchbase API.

3. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```


## Running the pipeline

Run the entire process with one command. Provide the CSV of technologies and the
year range for the papers you want to fetch:

```bash
python pipeline/run_pipeline.py \
  --tech-csv pipeline/techlist.csv \
  --start-year 2019 --end-year 2024
```

- If you set `CRUNCHBASE_API_KEY=NO_KEY`, the pipeline will use the company snapshot data from 1 July 2025 (`pipeline/company-data/crunchbase_fetching_raw.json`).
- If you provide a valid Crunchbase API key, the pipeline will fetch the latest company data from Crunchbase as before.

The resulting Neo4j import file will be written to
`pipeline/neo4j-files`.

## Import into Neo4j

**Import Node Files**
Importing the node files in neo4j is very straigt forward and easy in Neo4j Desktop.

1. Upload the files in the upload section highlighted by the red frame.

<img width="941" height="532" alt="image" src="https://github.com/user-attachments/assets/2fe748a5-2c6f-47ad-a34c-f7b490466060" />

2. Add a label and select the respective file to map. Map the properties from file
<img width="946" height="672" alt="image" src="https://github.com/user-attachments/assets/63450589-0877-485e-82f4-af760597920c" />

3. Hit "Run import" and repeat this step for Companies and Technologies. HOWEVER NOT FOR THE PAPERS
4. For the papers, upload the paper nodes into the import folder. Where to find this folder is specified here: [Default file locations - Operations Manual](https://neo4j.com/docs/operations-manual/current/configuration/file-locations/#neo4j-import)
5. Execute this cypher query:
```cypher
LOAD CSV WITH HEADERS FROM "file:///papers_nodes.csv" AS row
CREATE (p:Paper {
  paper_id:    row.paper_id,
  link:        row.link,
  title:       row.title,
  authors:     coalesce(split(row.authors, ";"), []),
  pub_date:    date(row.pub_date),
  institutions: coalesce(split(row.institution, ";"), [])
});
```
This is because we need to have authors and institutions in an array. 

**Importing Relationships**
This step requires entering cypher queries and uploading the link files to a specific folder.
1. Upload link files to import folder of your database. Where to find this folder is specified here: [Default file locations - Operations Manual](https://neo4j.com/docs/operations-manual/current/configuration/file-locations/#neo4j-import)
2. Execute these cypher queries:
```cypher
LOAD CSV WITH HEADERS FROM 'file:///company_technology_links.csv' AS row

// Match existing company and technology nodes
MATCH (c:Company {company_id: row.company_id})
MATCH (t:Technology {tech_name: row.technology_name})

// Create relationship
MERGE (c)-[:USES]->(t);
```
```cypher
LOAD CSV WITH HEADERS FROM 'file:///paper_technology_links.csv' AS row

// Match existing company and technology nodes
MATCH (p:Paper {paper_id: row.paper_id})
MATCH (t:Technology {tech_name: row.technology_name})

// Create relationship
MERGE (p)-[:TALKS_ABOUT]->(t);
```
The company-paper link file is optional as it is just the transitive linking of paper-technology -> technology-company

## Knowledge Graph Schema

### Entities

---

#### 1. Company

| Property            | Type       | Description               |
|--------------------|------------|--------------------------|
| company_id (id)    | string     |               id - crunchbase permalink      |
| company_name       | string     |         name of the company     |
| founding_date      | datetime   |       founding date of the company                   |
| location           | string     |location where the company is situated    |
| funding_total_usd  | float      |                    total funding the company in USD $      |
| funding_stage      | string     |             funding stage (late stage venture, early stage venture)             |

---

#### 2. Paper

| Property            | Type       | Description               |
|--------------------|------------|--------------------------|
| paper_id           | string     |           OpenAlex ID               |
| link               | string (URL)|            link to OpenAlex Page              |
| title              | string     |               Title of the Paper           |
| authors            | array      |authors of the paper       |
| pub_date           | datetime   |publication date           |
| institutions       | array      |institutions that are affiliated with any author of the paper              |

---

#### 3. Technology

| Property            | Type       | Description               |
|--------------------|------------|--------------------------|
| tech_name          | string     |name of the technology                          |
| description        | string     |short description of the technology    |

---

### Relationships

- **Company → USES → Technology**
- **Paper → TALKS_ABOUT → Technology**



