import os

# Step 1: Scrape research papers
os.system("python scripts/scrapex.py")

# Step 2: Clean data
os.system("python scripts/cleaner.py --input data/raw_research_papers.csv --output data/cleaned_data.csv")

# Step 3: Perform analysis
os.system("python scripts/Pandas_Analysis.py")

# Step 4: Generate visualizations
os.system("python scripts/Visualization.py")

# Step 5: Compute text representations
os.system("python scripts/Text_Representation.py")
