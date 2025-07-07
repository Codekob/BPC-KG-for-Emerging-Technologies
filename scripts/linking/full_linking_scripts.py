#!/usr/bin/env python3
"""
run_linking_pipeline.py

Simplified script that executes all linking steps sequentially:
  1) classify_papers.py
  2) link_papers_to_technologies.py
  3) link_companies_to_technologies.py
  4) link_companies_to_papers.py

Just run:
    python3 run_linking_pipeline.py

And it will stop on the first failure.
"""
import subprocess
import sys

SCRIPTS = [
    #'../classify/classify_papers_v2.py',
    'link_papers_to_technology.py',
    'link_comp_to_technology.py',
    'link_comp_to_paper.py'
]

def main():
    for script in SCRIPTS:
        print(f"Running {script}...")
        result = subprocess.run(['python3', script])
        if result.returncode != 0:
            print(f"Error: {script} exited with code {result.returncode}", file=sys.stderr)
            sys.exit(result.returncode)
    print("Pipeline completed successfully.")

if __name__ == '__main__':
    main()

