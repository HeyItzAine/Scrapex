import requests
import csv
import argparse
import os
import time
import math
import logging

# --- PROMETHEUS IMPORT ---
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# --- PROMETHEUS METRICS ---
API_REQUESTS = Counter("semantic_api_requests_total", "Total API requests made", ["type", "status"])
API_REQUEST_DURATION = Histogram("semantic_api_request_duration_seconds", "Time taken per API request", ["type"])
TOTAL_PAPERS_SCRAPED = Gauge("semantic_papers_scraped_total", "Total papers scraped and saved to CSV")
LAST_SCRAPE_TIME = Gauge("semantic_last_scrape_unixtime", "Unix time of the last successful scrape")

# --- LOGGING CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
FIELDS = "title,abstract,authors,venue,journal,year,externalIds"
HEADERS = {"User-Agent": "SemanticScholarScraper/1.0"}

def exponential_backoff_sleep(attempt, base_delay=1, max_delay=60):
    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
    logger.info(f"Waiting for {delay:.2f} seconds before retrying...")
    time.sleep(delay)

def fetch_paper_ids(query, total_limit_per_query=10000, per_page=1000, initial_delay=1, max_retries=5):
    next_token = None
    query_ids = set()
    logger.info(f"Starting to fetch paper IDs for query: '{query}' (Target: {total_limit_per_query} IDs)")

    while len(query_ids) < total_limit_per_query:
        attempt = 0
        while attempt < max_retries:
            attempt += 1
            params = {
                "query": query,
                "limit": per_page,
                "fields": "paperId"
            }
            if next_token:
                params["token"] = next_token

            logger.info(f"Fetching page for '{query}' with token: {next_token or 'start'} (Attempt {attempt})")

            try:
                start_time = time.time()
                response = requests.get(SEARCH_URL, headers=HEADERS, params=params)
                duration = time.time() - start_time
                API_REQUEST_DURATION.labels(type="fetch_id").observe(duration)

                if response.status_code == 429:
                    API_REQUESTS.labels(type="fetch_id", status="failure").inc()
                    exponential_backoff_sleep(attempt, initial_delay)
                    continue

                response.raise_for_status()
                API_REQUESTS.labels(type="fetch_id", status="success").inc()

                resp_json = response.json()
                data = resp_json.get("data", [])
                next_token = resp_json.get("next")  # note: bulk may not paginate beyond first page
                ids_page = {paper["paperId"] for paper in data if "paperId" in paper}

                query_ids.update(ids_page)
                logger.info(f"Fetched {len(ids_page)} IDs. Total: {len(query_ids)}/{total_limit_per_query}")

                break  # bulk endpoint only returns one page

            except requests.RequestException as e:
                API_REQUESTS.labels(type="fetch_id", status="failure").inc()
                logger.error(f"Error fetching IDs: {e}")
                exponential_backoff_sleep(attempt, initial_delay)

        break  # exit after one bulk page fetch

    return set(list(query_ids)[:total_limit_per_query])

def fetch_metadata_batch(paper_ids, initial_delay=1, max_retries=5):
    all_metadata = []
    batch_size = 100
    total_batches = math.ceil(len(paper_ids) / batch_size)
    paper_ids_list = list(paper_ids)

    for i in range(total_batches):
        batch = paper_ids_list[i * batch_size:(i + 1) * batch_size]
        attempt = 0

        while attempt < max_retries:
            attempt += 1
            try:
                logger.info(f"Fetching metadata batch {i+1}/{total_batches} (Attempt {attempt})")
                start_time = time.time()
                response = requests.post(
                    BATCH_URL,
                    headers=HEADERS,
                    json={"ids": batch},
                    params={"fields": FIELDS}
                )
                duration = time.time() - start_time
                API_REQUEST_DURATION.labels(type="fetch_metadata").observe(duration)

                if response.status_code == 429:
                    API_REQUESTS.labels(type="fetch_metadata", status="failure").inc()
                    exponential_backoff_sleep(attempt, initial_delay)
                    continue

                response.raise_for_status()
                API_REQUESTS.labels(type="fetch_metadata", status="success").inc()

                papers = response.json()
                valid_papers = [p for p in papers if p and isinstance(p, dict) and p.get("paperId")]

                for paper in valid_papers:
                    authors_list = [a["name"] for a in paper.get("authors", [])]
                    journal = paper.get("journal")
                    journal_name = journal.get("name", "N/A") if journal else "N/A"
                    publisher_name = journal.get("publisher", "N/A") if journal else "N/A"
                    doi = paper.get("externalIds", {}).get("DOI", "N/A")

                    all_metadata.append({
                        "title": paper.get("title", "N/A"),
                        "abstract": paper.get("abstract", "N/A"),
                        "authors": "; ".join(authors_list),
                        "venue": paper.get("venue") or journal_name,
                        "publisher": publisher_name,
                        "year": paper.get("year", "N/A"),
                        "doi": doi
                    })
                break
            except requests.RequestException as e:
                API_REQUESTS.labels(type="fetch_metadata", status="failure").inc()
                logger.error(f"Error fetching metadata batch {i+1}: {e}")
                exponential_backoff_sleep(attempt, initial_delay)
    return all_metadata

def save_to_csv(papers, output_file):
    if not papers:
        return
    try:
        with open(output_file, mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["Title", "Abstract", "Authors", "Venue", "Publisher", "Year", "DOI"])
            for p in papers:
                writer.writerow([p["title"], p["abstract"], p["authors"], p["venue"], p["publisher"], p["year"], p["doi"]])
        TOTAL_PAPERS_SCRAPED.set(len(papers))
        LAST_SCRAPE_TIME.set_to_current_time()
        logger.info(f"Saved {len(papers)} papers to {output_file}")
    except IOError as e:
        logger.error(f"CSV write error: {e}")

def main():
    start_http_server(9000)
    logger.info("Prometheus metrics server running at http://localhost:9000")

    parser = argparse.ArgumentParser(description="Semantic Scholar Metadata Scraper")
    parser.add_argument("--queries", type=str, required=True,
                        help="Comma-separated list of search queries or path to file with queries")
    parser.add_argument("--total_per_query", type=int, default=10000,
                        help="Total number of results to fetch for EACH query (default: 10000)")
    parser.add_argument("--initial_delay", type=int, default=1,
                        help="Initial delay between successful requests in seconds")
    parser.add_argument("--max_retries", type=int, default=5,
                        help="Maximum number of retries for API requests before skipping batch")

    args = parser.parse_args()

    if os.path.exists(args.queries):
        with open(args.queries, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
    else:
        queries = [q.strip() for q in args.queries.split(',') if q.strip()]

    script_directory = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_directory)
    data_dir = os.path.join(project_root, "Data")
    os.makedirs(data_dir, exist_ok=True)
    output_file = os.path.join(data_dir, "semantic_combined_bulk_data.csv")

    all_ids = set()
    for q in queries:
        ids = fetch_paper_ids(q, total_limit_per_query=args.total_per_query,
                              initial_delay=args.initial_delay, max_retries=args.max_retries)
        all_ids.update(ids)

    if all_ids:
        metadata = fetch_metadata_batch(all_ids,
                                        initial_delay=args.initial_delay,
                                        max_retries=args.max_retries)
        save_to_csv(metadata, output_file)

if __name__ == "__main__":
    main()