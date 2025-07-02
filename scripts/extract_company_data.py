#!/usr/bin/env python3
"""
Extract company name and description from Crunchbase DataFrame and save to CSV.
"""

import pandas as pd
import os

def extract_company_data(df, output_file='../data/company-data/companies_with_descriptions.csv'):
    """
    Extract identifier.value and short_description columns from DataFrame and save to CSV.
    
    Args:
        df: DataFrame with Crunchbase company data
        output_file: Path to save the CSV file
    """
    # Extract only the identifier.value and short_description columns
    company_data = df[['identifier.value', 'short_description']].copy()
    
    # Rename columns for clarity
    company_data.columns = ['company_name', 'description']
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to CSV
    company_data.to_csv(output_file, index=False)
    
    print(f"Saved {len(company_data)} companies to {output_file}")
    print("\nFirst few entries:")
    print(company_data.head())
    
    return company_data

if __name__ == "__main__":
    # Example usage - you would run this after fetching data in your notebook
    print("This script should be imported and used in your notebook after fetching company data.")
    print("Usage: extract_company_data(df)") 