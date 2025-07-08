# linking done through technologies


import pandas as pd

# load your two link files
papers = pd.read_csv('pipeline/neo4j-files/paper_technology_links.csv')        # cols: paper_id, technology_name
companies = pd.read_csv('pipeline/neo4j-files/company_technology_links.csv')   # cols: company_id, technology_name


# inner-join on technology_name
company_paper = companies.merge(
    papers,
    on='technology_name',
    how='inner'
)[['company_id','paper_id','technology_name']]

# drop duplicates if you only want unique company↔paper pairs
company_paper = company_paper.drop_duplicates(subset=['company_id','paper_id'])

# write it out
company_paper.to_csv('pipeline/neo4j-files/company_paper_links.csv', index=False)
