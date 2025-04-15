import os

# Step 1: Scrape research papers
os.system("python scripts/scrapex.py")

os.system('python scripts/Semantic_Scrapex.py --query "research OR review OR paper" --total 1000 --per_page 100 --delay 1')

# Step 2: Clean data
os.system("python scripts/cleaner.py --input data/research_titles.csv --output data/research_titles_cleaned.csv")

os.system("python scripts/cleaner.py --input data/semantic_titles.csv --output data/semantic_titles_cleaned.csv")

# Step 3: Perform analysis
os.system("python scripts/Pandas_Analysis.py")

# Step 4: Generate visualizations
os.system("python scripts/Visualization.py")

# Step 5: Compute text representations
os.system("python scripts/Text_Representation.py")

# Step 6: Perform topic modeling
os.system("python scripts/BERTopic.py data/research_titles_cleaned.csv all")
os.system("python scripts/BERTopic.py data/semantic_titles_cleaned.csv all")
