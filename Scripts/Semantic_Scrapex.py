import requests
import csv
import argparse
import os
import time
import math
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"

FIELDS = "title,abstract,authors,venue,journal,year,externalIds"

HEADERS = {
    "User-Agent": "SemanticScholarScraper/1.0",
    # OPTIONAL: If you have a Semantic Scholar API key, uncomment the line below
    # and replace "YOUR_API_KEY_HERE" with your actual key for higher rate limits.
    # "x-api-key": "YOUR_API_KEY_HERE"
}

# --- Helper Function for Exponential Backoff ---
def exponential_backoff_sleep(attempt, base_delay=1, max_delay=60):
    """
    Calculates and sleeps for an exponential backoff delay.
    Increases the delay with each retry attempt to avoid overwhelming the API.
    """
    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
    logger.info(f"Waiting for {delay:.2f} seconds before retrying...")
    time.sleep(delay)

# --- Fetch Paper IDs using Bulk Search ---
def fetch_paper_ids(query, total_limit_per_query=1000, per_page=1000, initial_delay=1, max_retries=5):
    """
    Fetches paper IDs for a given query from the Semantic Scholar API using the /bulk endpoint.
    Handles rate limits and other request errors with exponential backoff.
    
    Args:
        query (str): The search query.
        total_limit_per_query (int): The maximum number of paper IDs to fetch for THIS SPECIFIC QUERY.
        per_page (int): The number of results to request per page (bulk endpoint returns 1000 per page).
        initial_delay (int): Initial delay for exponential backoff.
        max_retries (int): Maximum number of retries for an API request.
        
    Returns:
        set: A set of paper IDs (to automatically handle duplicates from API's side).
    """
    next_token = None # Pagination uses 'token' instead of 'offset' for bulk search
    query_ids = set() # Use a set to store IDs for current query to avoid duplicates
    
    logger.info(f"Starting to fetch paper IDs for query: '{query}' (Target: {total_limit_per_query} IDs) using bulk search.")

    while len(query_ids) < total_limit_per_query:
        attempt = 0 # Reset attempt for each new page/token

        while attempt < max_retries:
            attempt += 1
            params = {
                "query": query,
                "limit": per_page, # The API will still return up to 1000 per page
                "fields": "paperId" # Only fetching paperId in this step
            }
            if next_token:
                params["token"] = next_token

            logger.info(f"Fetching page for '{query}' with token: {next_token if next_token else 'start'} (Attempt {attempt}/{max_retries})...")

            try:
                response = requests.get(SEARCH_URL, headers=HEADERS, params=params)

                if response.status_code == 429:
                    logger.warning("Rate limit exceeded while fetching IDs. Retrying with backoff...")
                    exponential_backoff_sleep(attempt, initial_delay)
                    continue # Retry the same request with the same token

                response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
                response_json = response.json()
                data = response_json.get("data", [])
                next_token = response_json.get("next", None) # Get the token for the next page

                ids_page = {paper["paperId"] for paper in data if "paperId" in paper} # Use set for page IDs

                if not ids_page and not next_token:
                    logger.info(f"No more paper IDs found for query '{query}' or token.")
                    break # No more results, break the inner retry loop

                new_ids_count = len(ids_page - query_ids) # Count truly new IDs from this page
                query_ids.update(ids_page)
                
                logger.info(f"Fetched {len(ids_page)} IDs on this page ({new_ids_count} new). Total IDs for '{query}': {len(query_ids)}/{total_limit_per_query}...")
                
                # If we've reached or exceeded the total_limit_per_query, we can stop early
                if len(query_ids) >= total_limit_per_query:
                    break 

                # Reset attempt counter for next page/token
                attempt = 0 
                time.sleep(initial_delay) # Small delay between successful requests for different pages

                if not next_token: # No more tokens means no more results
                    logger.info(f"Reached the end of available bulk search results for query '{query}'.")
                    break # Break the inner retry loop and outer while loop

            except requests.RequestException as e:
                logger.error(f"Request error during ID fetch for '{query}' (page with token {next_token if next_token else 'start'}): {e}")
                if attempt < max_retries:
                    logger.warning(f"Retrying page after backoff (attempt {attempt}/{max_retries})...")
                    exponential_backoff_sleep(attempt, initial_delay)
                else:
                    logger.critical(f"Max retries ({max_retries}) exceeded for current page of query '{query}'. Skipping remaining IDs for this query.")
                    break # Break inner retry loop if max retries are hit

        if attempt == max_retries and len(query_ids) < total_limit_per_query:
            logger.warning(f"Could not complete fetching all desired IDs for query '{query}' due to repeated errors.")
            break # Break outer while loop if max retries for a page were hit

        if not next_token and len(query_ids) < total_limit_per_query: # If we broke because no next token, but haven't hit total_limit
            break # Exit the outer loop

    logger.info(f"Finished fetching paper IDs for query '{query}'. Total unique IDs collected: {len(query_ids)}")
    # Ensure we don't return more than the total_limit_per_query if the last page overshoots
    return set(list(query_ids)[:total_limit_per_query])


# --- Fetch Metadata in Batches ---
def fetch_metadata_batch(paper_ids, initial_delay=1, max_retries=5):
    """
    Fetches detailed metadata for a list of paper IDs in batches from Semantic Scholar API.
    Handles rate limits and other request errors with exponential backoff.
    """
    all_metadata = []
    batch_size = 100 # Semantic Scholar batch API limit for POST requests is up to 500
    total_batches = math.ceil(len(paper_ids) / batch_size)

    logger.info(f"\nStarting to fetch metadata for {len(paper_ids)} papers in {total_batches} batches.")

    # Convert paper_ids set to a list for slicing
    paper_ids_list = list(paper_ids) 

    for i in range(total_batches):
        batch = paper_ids_list[i * batch_size:(i + 1) * batch_size]
        attempt = 0 
        batch_successful = False

        while not batch_successful and attempt < max_retries:
            attempt += 1
            try:
                logger.info(f"Processing batch {i+1}/{total_batches} ({len(batch)} papers) (Attempt {attempt}/{max_retries})...")
                response = requests.post(
                    BATCH_URL,
                    headers=HEADERS,
                    json={"ids": batch},
                    params={"fields": FIELDS}
                )

                if response.status_code == 429:
                    logger.warning("Rate limit hit while fetching metadata. Retrying with backoff...")
                    exponential_backoff_sleep(attempt, initial_delay)
                    continue # Retry the same batch

                response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
                papers = response.json()

                # CRITICAL FIX: Filter out None or empty paper objects from the response
                # The batch endpoint can return nulls if an ID is not found or invalid
                valid_papers = [p for p in papers if p is not None and isinstance(p, dict) and p.get("paperId")]

                for paper in valid_papers: # Iterate only over valid paper objects
                    # Safely get values, providing 'N/A' as default for missing data

                    authors_list = [a["name"] for a in paper.get("authors", [])]
                    journal_data = paper.get("journal")
                    journal_name = journal_data.get("name", "N/A") if journal_data else "N/A"
                    publisher_name = journal_data.get("publisher", "N/A") if journal_data else "N/A"
                    doi_value = paper.get("externalIds", {}).get("DOI", "N/A")

                    all_metadata.append({
                        "title": paper.get("title", "N/A"),
                        "abstract": paper.get("abstract", "N/A"),
                        "authors": "; ".join(authors_list),
                        "venue": paper.get("venue") or journal_name,
                        "publisher": publisher_name,
                        "year": paper.get("year", "N/A"),
                        "doi": doi_value
                    })
                batch_successful = True # Mark batch as successful
                logger.info(f"Successfully processed batch {i+1}/{total_batches}.")
                time.sleep(initial_delay) # Small delay between successful batches

            except requests.RequestException as e:
                logger.error(f"Error during batch fetch {i+1}: {e}")
                if attempt < max_retries:
                    logger.warning(f"Retrying batch after backoff (attempt {attempt}/{max_retries})...")
                    exponential_backoff_sleep(attempt, initial_delay)
                else:
                    logger.critical(f"Max retries ({max_retries}) exceeded for batch {i+1}. Skipping this batch.")

    logger.info(f"\nFinished fetching metadata. Total papers with metadata: {len(all_metadata)}")
    return all_metadata

# --- Save to CSV ---
def save_to_csv(papers, output_file):
    """Saves the fetched paper metadata to a CSV file."""
    if not papers:
        logger.warning("No papers to save. CSV file will not be created.")
        return

    csv_headers = ["Title", "Abstract", "Authors", "Venue", "Publisher", "Year", "DOI"]

    try:
        with open(output_file, mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)

            for paper in papers:
                writer.writerow([
                    paper["title"],
                    paper["abstract"],
                    paper["authors"],
                    paper["venue"],
                    paper["publisher"],
                    paper["year"],
                    paper["doi"]
                ])
        logger.info(f"\n✅ Saved {len(papers)} papers to {output_file}")
    except IOError as e:
        logger.error(f"Error writing to CSV file {output_file}: {e}")

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Semantic Scholar Metadata Scraper")
    parser.add_argument("--queries", type=str, required=True,
                        help="Comma-separated list of search queries (e.g., 'query1,query2'). "
                             "Alternatively, provide a path to a text file with one query per line.")
    parser.add_argument("--total_per_query", type=int, default=1000,
                        help="Total number of results to fetch for EACH query")
    parser.add_argument("--initial_delay", type=int, default=1,
                         help="Initial delay between successful requests in seconds (used in exponential backoff)")
    parser.add_argument("--max_retries", type=int, default=5,
                         help="Maximum number of retries for API requests before giving up on a specific call/batch")

    args = parser.parse_args()

    # Determine if queries come from a file or a string
    queries = []
    if os.path.exists(args.queries):
        logger.info(f"Reading queries from file: {args.queries}")
        with open(args.queries, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
    else:
        logger.info(f"Using queries from command line: {args.queries}")
        queries = [q.strip() for q in args.queries.split(',') if q.strip()]

    if not queries:
        logger.error("No queries provided. Please specify queries via --queries argument.")
        return

    # --- Path Adjustment for Data Folder ---
    script_directory = os.path.dirname(os.path.abspath(__file__))
    project_root_directory = os.path.dirname(script_directory)
    DATA_DIR = os.path.join(project_root_directory, "Data")
    # --- End Path Adjustment ---

    os.makedirs(DATA_DIR, exist_ok=True)
    output_file = os.path.join(DATA_DIR, "semantic_combined_bulk_data.csv") # Updated output file name

    all_unique_paper_ids = set() # Use a set to collect all unique IDs across all queries

    for i, query in enumerate(queries):
        logger.info(f"\n--- Processing Query {i+1}/{len(queries)}: '{query}' ---")
        current_query_ids = fetch_paper_ids(
            query=query,
            total_limit_per_query=args.total_per_query,
            initial_delay=args.initial_delay,
            max_retries=args.max_retries
        )
        all_unique_paper_ids.update(current_query_ids)
        logger.info(f"Unique IDs collected so far: {len(all_unique_paper_ids)}")

    if all_unique_paper_ids:
        logger.info(f"\nTotal unique paper IDs collected from all queries: {len(all_unique_paper_ids)}")
        metadata = fetch_metadata_batch(
            all_unique_paper_ids, # Pass the set of all unique IDs
            initial_delay=args.initial_delay,
            max_retries=args.max_retries
        )
        save_to_csv(metadata, output_file)
    else:
        logger.warning("No unique paper IDs were fetched from any query. Exiting without creating a CSV file.")


if __name__ == "__main__":
    main()