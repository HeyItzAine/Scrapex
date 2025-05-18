import time
import random
import csv
import argparse
import unicodedata
import requests
from bs4 import BeautifulSoup
import logging
import os
from dotenv import load_dotenv
from prometheus_client import Counter, Histogram, Gauge, start_http_server

load_dotenv("../apitoken.env")
APIKEY = os.getenv("SERPAPIKEY")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

REQUEST_COUNT = Counter(
    'scrapex_requests_total',
    'Total number of scrape requests',
    ['status']  # label: 'success' or 'failure'
)

REQUEST_DURATION = Histogram(
    'scrapex_request_duration_seconds',
    'Histogram of scrape request durations'
)

EXCEPTION_COUNT = Counter(
    'scrapex_exceptions_total',
    'Total exceptions raised during scraping'
)

LAST_SCRAPE_TIMESTAMP = Gauge(
    'scrapex_last_scrape_unixtime',
    'Unix timestamp of the last scrape executed'
)

SCRAPEX_TOTAL_SCRAPE_TIME = Gauge(
    'scrapex_total_scrape_time_seconds',
    'Total time taken for the last full scrape (all pages)'
)

logger = logging.getLogger(__name__)

class GoogleScholarScraperRequests:
    """A class to scrape research paper titles and authors from Google Scholar using requests with multithreading."""
    
    def __init__(self, 
                 user_agents=None, 
                 output_file="research_titles.csv",
                 max_pages=5,
                 delay_range=(2, 5)):
        """
        Initialize the scraper.
        """
        self.output_file = output_file
        self.max_pages = max_pages
        self.delay_range = delay_range
        
        # Default user agents if none provided
        self.user_agents = user_agents or [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
        ]
        
        # Initialize collected papers list
        self.collected_papers = []
        # Bind the total scrape time gauge to the instance
        self.SCRAPEX_TOTAL_SCRAPE_TIME = SCRAPEX_TOTAL_SCRAPE_TIME
    
    def random_delay(self):
        """Sleep for a random amount of time."""
        delay = random.uniform(*self.delay_range)
        logger.debug(f"Waiting for {delay:.2f} seconds")
        time.sleep(delay)
    
    def extract_papers(self, html_content):
        """
        Extract research titles, authors, and links from HTML content using robust selectors (adapted to Google Scholar structure as seen in MHTML).
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        papers = []
        # Each result block is in a div with class 'gs_r gs_or gs_scl'
        result_blocks = soup.find_all("div", class_="gs_r")
        for block in result_blocks:
            # The main info is inside the gs_ri div
            info = block.find("div", class_="gs_ri")
            if not info:
                continue
            # Title and link
            title_elem = info.find("h3", class_="gs_rt")
            if title_elem:
                link_elem = title_elem.find("a")
                title = title_elem.get_text(strip=True)
                link = link_elem["href"] if link_elem else None
            else:
                title = None
                link = None
            # Authors
            authors_elem = info.find("div", class_="gs_a")
            authors = authors_elem.get_text(strip=True) if authors_elem else None
            if title:
                papers.append({"title": title, "authors": authors, "link": link})
        return papers
    
    def save_to_csv(self):
        """Save the collected papers to a CSV file with UTF-8 encoding."""
        try:
            with open(self.output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Title', 'Authors'])  # Header
                for paper in self.collected_papers:
                    writer.writerow([paper['title'], paper['authors']])
            logger.info(f"Successfully saved {len(self.collected_papers)} papers to {self.output_file}")
        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")
            raise
        
    def fetch_page(self, page_number, base_url):
        
        url = base_url if page_number == 0 else f"{base_url}&start={page_number * 10}"
        headers = {"User-Agent": random.choice(self.user_agents)}
        logger.info(f"Fetching page {page_number + 1}: {url}")
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                logger.warning(f"Non-200 status code received on page {page_number + 1}: {response.status_code}")
                return []
        except requests.RequestException as e:
            logger.error(f"Request failed on page {page_number + 1}: {e}")
            return []

        self.random_delay()
        papers = self.extract_papers(response.text)
        if not papers:
            logger.warning(f"First 500 chars of HTML for page {page_number + 1}:\n{response.text[:500]}")
        return papers
    
    def fetch_page_serpapi(self, query, page_number, api_key):
        """
        Fetch a page of Google Scholar results using SerpApi.
        """
        params = {
            "engine": "google_scholar",
            "q": query,
            "start": page_number * 10,
            "api_key": api_key
        }
        url = "https://serpapi.com/search"
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            results = data.get("organic_results", [])
            papers = []
            for result in results:
                title = result.get("title")
                link = result.get("link")
                authors = result.get("publication_info", {}).get("summary")
                papers.append({"title": title, "authors": authors, "link": link})
            return papers
        except Exception as e:
            logger.error(f"SerpApi error on page {page_number + 1}: {e}")
            return []

    def scrape(self, broad_query="research OR review OR paper", use_serpapi=False, serpapi_key=None):
        """
        Main method to scrape Google Scholar for research titles and authors.
        If use_serpapi is True and serpapi_key is provided, use SerpApi for scraping.
        Prometheus metrics are updated for each page: request count, duration, errors, and last scrape time.
        """
        formatted_query = broad_query.replace(' ', '+')
        total_scrape_start = time.time()
        success_count = 0
        fail_count = 0
        exception_count = 0
        if use_serpapi and serpapi_key:
            for page_number in range(self.max_pages):
                page_start = time.time()
                try:
                    page_papers = self.fetch_page_serpapi(broad_query, page_number, serpapi_key)
                    duration = time.time() - page_start
                    REQUEST_DURATION.observe(duration)
                    if not page_papers:
                        logger.warning(f"No papers found on page {page_number + 1} (SerpApi). It may have been blocked or reached the end.")
                        REQUEST_COUNT.labels(status='failure').inc()
                        fail_count += 1
                    else:
                        logger.info(f"Extracted {len(page_papers)} papers from page {page_number + 1} (SerpApi)")
                        REQUEST_COUNT.labels(status='success').inc()
                        success_count += 1
                    self.collected_papers.extend(page_papers)
                    LAST_SCRAPE_TIMESTAMP.set_to_current_time()
                    time.sleep(random.uniform(1, 2))  # polite delay
                except Exception as exc:
                    logger.error(f"Page {page_number + 1} generated an exception (SerpApi): {exc}")
                    EXCEPTION_COUNT.inc()
                    REQUEST_COUNT.labels(status='failure').inc()
                    exception_count += 1
            total_duration = time.time() - total_scrape_start
            # Custom gauge for total scrape time
            if hasattr(self, 'SCRAPEX_TOTAL_SCRAPE_TIME'):
                self.SCRAPEX_TOTAL_SCRAPE_TIME.set(total_duration)
            self.save_to_csv()
            return self.collected_papers
        else:
            base_url = f"https://scholar.google.com/scholar?q={formatted_query}"
            for page_number in range(self.max_pages):
                page_start = time.time()
                try:
                    page_papers = self.fetch_page(page_number, base_url)
                    duration = time.time() - page_start
                    REQUEST_DURATION.observe(duration)
                    if not page_papers:
                        logger.warning(f"No papers found on page {page_number + 1}. It may have been blocked or reached the end.")
                        REQUEST_COUNT.labels(status='failure').inc()
                        fail_count += 1
                    else:
                        logger.info(f"Extracted {len(page_papers)} papers from page {page_number + 1}")
                        REQUEST_COUNT.labels(status='success').inc()
                        success_count += 1
                    self.collected_papers.extend(page_papers)
                    LAST_SCRAPE_TIMESTAMP.set_to_current_time()
                    time.sleep(random.uniform(20, 30))
                except Exception as exc:
                    logger.error(f"Page {page_number + 1} generated an exception: {exc}")
                    EXCEPTION_COUNT.inc()
                    REQUEST_COUNT.labels(status='failure').inc()
                    exception_count += 1
            total_duration = time.time() - total_scrape_start
            if hasattr(self, 'SCRAPEX_TOTAL_SCRAPE_TIME'):
                self.SCRAPEX_TOTAL_SCRAPE_TIME.set(total_duration)
            self.save_to_csv()
            return self.collected_papers



def main():
    parser = argparse.ArgumentParser(description='Scrape research paper titles and authors from Google Scholar using requests with multithreading')
    # Dynamically find the Data/ directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up from Scripts/
    DATA_DIR = os.path.join(BASE_DIR, "Data")  # Points to Scrapex/Data

    # Ensure the Data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    parser.add_argument('--output', type=str, default=os.path.join(DATA_DIR, "research_titles.csv"))

    parser.add_argument('--pages', type=int, default=10,
                        help='Maximum number of pages to scrape')
    parser.add_argument('--query', type=str, 
                        default='research OR review OR paper',
                        help='Broad query to use for getting results')
    parser.add_argument('--serpapi', type=str, default=None, help='SerpApi API key (if provided, will use SerpApi for scraping)')
    
    args = parser.parse_args()
    
    scraper = GoogleScholarScraperRequests(
        output_file=args.output,
        max_pages=args.pages
    )

    try:
        start_http_server(8000)  # Expose Prometheus metrics on port 8000
        start_time = time.time()
        serpapi_key = args.serpapi or APIKEY
        use_serpapi = serpapi_key is not None
        papers = scraper.scrape(broad_query=args.query, use_serpapi=use_serpapi, serpapi_key=serpapi_key)
        duration = time.time() - start_time
        REQUEST_COUNT.labels(status='success').inc()
        REQUEST_DURATION.observe(duration)
        LAST_SCRAPE_TIMESTAMP.set_to_current_time()
        print(f"Successfully scraped {len(papers)} research papers.")
    except Exception as e:
        EXCEPTION_COUNT.inc()
        REQUEST_COUNT.labels(status='failure').inc()
        print(f"Error: {e}")
        return 1
    print("Scraping complete. Keeping metrics server alive for Prometheus...")

    # Keep the HTTP server alive so Prometheus can scrape
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("Exiting.")
    return 0

if __name__ == "__main__":
    exit(main())
