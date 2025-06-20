import pandas as pd

def analyze_data(input_files):
    for input_file in input_files:
        print(f"\nAnalyzing {input_file}...")
        df = pd.read_csv(input_file)

        print("Number of unique titles:", df["CleanedTitle"].nunique())
        print("Number of unique authors:", df["CleanedAuthors"].nunique())
        print("Average title length (words):", df["CleanedTitle"].apply(lambda x: len(x.split())).mean())

if __name__ == "__main__":
    analyze_data(["data/research_titles_cleaned.csv", "data/semantic_titles_cleaned.csv"])
