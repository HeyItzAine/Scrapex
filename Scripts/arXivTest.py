import argparse
import csv
import time
import random
import logging
import feedparser
from urllib.parse import urlencode
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

GROUP_NAME = "scrapex"
ARXIV_API_URL = "http://export.arxiv.org/api/query"

class ArxivScraper:
    def __init__(self, query, max_results=100, delay_range=(1, 3), output_file="arxiv_papers.csv", max_workers=4):
        self.query = query
        self.max_results = max_results
        self.delay_range = delay_range
        self.output_file = output_file
        self.max_workers = max_workers
        self.collected = []
        self._lock = threading.Lock()  # Thread-safe collection

    def random_delay(self):
        delay = random.uniform(*self.delay_range)
        logger.debug(f"Sleeping for {delay:.2f} seconds")
        time.sleep(delay)

    def fetch(self, start_index=0, batch_size=100):
        params = {
            'search_query': f'all:{self.query}',
            'start': start_index,
            'max_results': batch_size,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        url = f"{ARXIV_API_URL}?{urlencode(params)}"
        logger.info(f"Fetching entries {start_index + 1} to {start_index + batch_size}")
        feed = feedparser.parse(url)
        
        # Check for API errors or limitations
        if hasattr(feed, 'status') and feed.status != 200:
            logger.error(f"API returned status {feed.status}")
            return []
            
        # Get total results if available
        total_results = int(feed.feed.opensearch_totalresults) if hasattr(feed.feed, 'opensearch_totalresults') else None
        if total_results is not None:
            logger.info(f"Total available results: {total_results}")
            self.max_results = min(self.max_results, total_results)
            
        return feed.entries

    def extract_entry(self, entry):
        """Extract all relevant metadata from an arXiv entry."""
        # Basic metadata
        title = entry.title.strip()
        abstract = entry.summary.strip()
        authors = [author.name for author in entry.authors]
        
        # Publication details
        journal = getattr(entry, 'arxiv_journal_ref', None)
        publisher = 'arXiv'
        published = entry.published
        updated = getattr(entry, 'updated', None)
        year = published[:4]
        
        # Identifiers
        arxiv_id = entry.id.split('/abs/')[-1]
        doi = getattr(entry, 'arxiv_doi', None)
        
        # Links
        pdf_link = next((link.href for link in entry.links if link.type == 'application/pdf'), None)
        arxiv_link = next((link.href for link in entry.links if link.type == 'text/html'), None)
        
        # Categories
        primary_category = entry.tags[0]['term'] if hasattr(entry, 'tags') and entry.tags else None
        all_categories = [tag['term'] for tag in entry.tags] if hasattr(entry, 'tags') else []
        
        return {
            'title': title,
            'abstract': abstract,
            'authors': authors,
            'journal_conference_name': journal,
            'publisher': publisher,
            'year': year,
            'doi': doi,
            'group_name': GROUP_NAME,
            'arxiv_id': arxiv_id,
            'published_date': published,
            'updated_date': updated,
            'pdf_url': pdf_link,
            'arxiv_url': arxiv_link,
            'primary_category': primary_category,
            'categories': '; '.join(all_categories)
        }

    def add_papers(self, papers):
        """Thread-safe addition of papers to collection."""
        with self._lock:
            self.collected.extend(papers)

    def generate_year_queries(self):
        """Generate queries split by year ranges to get more results."""
        current_year = datetime.now().year
        start_year = current_year - 10  # Last 10 years
        queries = []
        
        for year in range(start_year, current_year + 1):
            year_query = f"({self.query}) AND submittedDate:[{year}0101 TO {year}1231]"
            queries.append((year_query, year))
        return queries

    def fetch_year(self, year_data):
        """Fetch all results for a specific year."""
        year_query, year = year_data
        year_papers = []
        start = 0
        batch_size = 100
        
        while True:
            params = {
                'search_query': year_query,
                'start': start,
                'max_results': batch_size,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            url = f"{ARXIV_API_URL}?{urlencode(params)}"
            logger.info(f"Fetching year {year}, entries {start + 1} to {start + batch_size}")
            
            feed = feedparser.parse(url)
            if hasattr(feed, 'status') and feed.status != 200:
                logger.error(f"API returned status {feed.status} for year {year}")
                break
                
            entries = feed.entries
            if not entries:
                break
                
            for entry in entries:
                paper = self.extract_entry(entry)
                year_papers.append(paper)
                
            if len(entries) < batch_size:
                break
                
            start += batch_size
            self.random_delay()
            
        logger.info(f"Year {year}: found {len(year_papers)} papers")
        return year_papers

    def scrape(self):
        """Scrape papers in parallel using year-based queries."""
        year_queries = self.generate_year_queries()
        total_papers = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_year = {executor.submit(self.fetch_year, year_data): year_data[1] 
                            for year_data in year_queries}
            
            for future in as_completed(future_to_year):
                year = future_to_year[future]
                try:
                    papers = future.result()
                    total_papers += len(papers)
                    self.add_papers(papers)
                    logger.info(f"Completed year {year}: {len(papers)} papers")
                    
                    if total_papers >= self.max_results:
                        for f in future_to_year:
                            f.cancel()
                        break
                except Exception as e:
                    logger.error(f"Year {year} generated an exception: {e}")
        
        # Trim to max_results if we got more
        if len(self.collected) > self.max_results:
            self.collected = self.collected[:self.max_results]
        
        self.save_csv()
        logger.info(f"Scraped {len(self.collected)} papers total.")
        return self.collected

    def save_csv(self):
        """Save collected papers to CSV with all metadata fields."""
        fieldnames = [
            'title', 'abstract', 'authors', 'journal_conference_name',
            'publisher', 'year', 'doi', 'group_name', 'arxiv_id',
            'published_date', 'updated_date', 'pdf_url', 'arxiv_url',
            'primary_category', 'categories'
        ]
        try:
            with open(self.output_file, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for paper in self.collected:
                    # Convert authors list to semicolon-separated string
                    paper['authors'] = '; '.join(paper['authors'])
                    writer.writerow(paper)
            logger.info(f"Data saved to {self.output_file}")
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Scrape papers from arXiv API in parallel')
    parser.add_argument('--query', type=str, required=True,
                        help='Search query for arXiv')
    parser.add_argument('--max_results', type=int, default=12000,
                        help='Total number of results to fetch')
    parser.add_argument('--output', type=str, default='arxiv_papers.csv',
                        help='Output CSV file path')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers')
    args = parser.parse_args()

    scraper = ArxivScraper(
        query=args.query,
        max_results=args.max_results,
        output_file=args.output,
        max_workers=args.workers
    )

    papers = scraper.scrape()
    print(f"Scraping complete: {len(papers)} papers saved to {args.output}")

if __name__ == '__main__':
    main()
