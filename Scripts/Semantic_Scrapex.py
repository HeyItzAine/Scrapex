import requests
import csv
import argparse
import os
import time

API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

def fetch_papers(query, total_limit=200, per_page=100, delay=1):
    headers = {"User-Agent": "SemanticScholarScraper/1.0"}
    offset = 0
    all_papers = []

    while len(all_papers) < total_limit:
        current_limit = min(per_page, total_limit - len(all_papers))
        params = {
            "query": query,
            "limit": current_limit,
            "offset": offset,
            "fields": "title,authors,abstract,url,paperId"
        }

        try:
            response = requests.get(API_URL, headers=headers, params=params)
            if response.status_code == 429:
                print("Rate limit exceeded. Retrying after delay...")
                time.sleep(delay)
                continue  # don't increase offset on retry

            response.raise_for_status()
            batch = response.json().get("data", [])
            print(f"Fetched {len(batch)} papers (offset {offset})")

            if not batch or len(batch) < current_limit:
                all_papers.extend(batch)
                break  # stop early if fewer than requested were returned

            all_papers.extend(batch)
            offset += current_limit
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            print("Retrying same batch after delay...")
            time.sleep(delay)

    return all_papers

def save_to_csv(papers, output_file):
    with open(output_file, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Title", "Authors", "Abstract", "URL", "Paper ID"])

        for paper in papers:
            title = paper.get("title", "N/A")
            authors = ", ".join([author["name"] for author in paper.get("authors", [])])
            abstract = paper.get("abstract", "N/A")
            url = paper.get("url", "N/A")
            paper_id = paper.get("paperId", "N/A")
            writer.writerow([title, authors, abstract, url, paper_id])

    print(f"Saved {len(papers)} papers to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Scrape research papers from Semantic Scholar")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "Data")
    os.makedirs(DATA_DIR, exist_ok=True)
    output_file = os.path.join(DATA_DIR, "semantic_titles.csv")

    parser.add_argument("--query", type=str, required=True, help="Search query")
    parser.add_argument("--total", type=int, default=200, help="Total number of results to fetch")
    parser.add_argument("--per_page", type=int, default=100, help="Results per API call (max 100)")
    parser.add_argument("--delay", type=int, default=1, help="Delay between retries in seconds")

    args = parser.parse_args()

    papers = fetch_papers(args.query, args.total, args.per_page, args.delay)
    save_to_csv(papers, output_file)

if __name__ == "__main__":
    main()
