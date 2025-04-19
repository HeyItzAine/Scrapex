import os

# Step 1: Scrape research papers
os.system("python scripts/scrapex.py")

os.system('python scripts/Semantic_Scrapex.py --query "research review paper" --total 1000 --per_page 50 /--delay 2')

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

# Step 7: Convert CSV to JSON
os.system("python scripts/Converter.py --csv semantic_titles_cleaned_BERTopic/semantic_titles_cleaned_with_topics.csv")
os.system("python scripts/Converter.py --csv research_titles_cleaned_BERTopic/research_titles_cleaned_with_topics.csv")