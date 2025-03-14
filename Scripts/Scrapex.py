import time
import random
import csv
import argparse
import unicodedata
import requests
from bs4 import BeautifulSoup
import logging
import concurrent.futures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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
    
    def random_delay(self):
        """Sleep for a random amount of time."""
        delay = random.uniform(*self.delay_range)
        logger.debug(f"Waiting for {delay:.2f} seconds")
        time.sleep(delay)
    
    def extract_papers(self, html_content):
        """
        Extract research titles and authors from HTML content using BeautifulSoup.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        papers = []
        
        title_elements = soup.select('h3.gs_rt')
        author_elements = soup.select('.gs_a')
        
        for title_elem, author_elem in zip(title_elements, author_elements):
            # Extract title
            citation_tag = title_elem.find('span', class_='gs_ctu')
            if citation_tag:
                citation_tag.decompose()
            
            title_link = title_elem.find('a')
            title = ' '.join(title_link.stripped_strings) if title_link else ' '.join(title_elem.stripped_strings)
            title = unicodedata.normalize("NFKC", title).replace('\xa0', ' ')
            
            # Extract authors (excluding journal and year information)
            author_text = author_elem.get_text()
            authors = author_text.split('-')[0].strip()
            
            if title:
                papers.append({"title": title, "authors": authors})
        
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
        """
        Fetch a single page and extract research papers.
        """
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
        return self.extract_papers(response.text)
    
    def scrape(self, broad_query="research OR review OR paper"):
        """
        Main method to scrape Google Scholar for research titles and authors using multithreading.
        """
        formatted_query = broad_query.replace(' ', '+')
        base_url = f"https://scholar.google.com/scholar?q={formatted_query}"
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_pages) as executor:
            future_to_page = {
                executor.submit(self.fetch_page, page_number, base_url): page_number 
                for page_number in range(self.max_pages)
            }
            for future in concurrent.futures.as_completed(future_to_page):
                page_number = future_to_page[future]
                try:
                    page_papers = future.result()
                    if not page_papers:
                        logger.warning(f"No papers found on page {page_number + 1}. It may have been blocked or reached the end.")
                    else:
                        logger.info(f"Extracted {len(page_papers)} papers from page {page_number + 1}")
                    self.collected_papers.extend(page_papers)
                except Exception as exc:
                    logger.error(f"Page {page_number + 1} generated an exception: {exc}")
        
        self.save_to_csv()
        return self.collected_papers


def main():
    parser = argparse.ArgumentParser(description='Scrape research paper titles and authors from Google Scholar using requests with multithreading')
    parser.add_argument('--output', type=str, default='../Data/research_titles.csv',
                        help='Output CSV file path')
    parser.add_argument('--pages', type=int, default=5,
                        help='Maximum number of pages to scrape')
    parser.add_argument('--query', type=str, 
                        default='research OR review OR paper',
                        help='Broad query to use for getting results')
    
    args = parser.parse_args()
    
    scraper = GoogleScholarScraperRequests(
        output_file=args.output,
        max_pages=args.pages
    )
    
    try:
        papers = scraper.scrape(broad_query=args.query)
        print(f"Successfully scraped {len(papers)} research papers.")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
