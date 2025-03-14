import time
import random
import csv
import argparse
import unicodedata
import requests
from bs4 import BeautifulSoup
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GoogleScholarScraperRequests:
    """A class to scrape research paper titles from Google Scholar using requests."""
    
    def __init__(self, 
                 user_agents=None, 
                 output_file="research_titles.csv",
                 max_pages=5,
                 delay_range=(2, 5)):
        """
        Initialize the scraper.
        
        Args:
            user_agents (list): List of user agents to rotate through
            output_file (str): Path to the output CSV file
            max_pages (int): Maximum number of pages to scrape
            delay_range (tuple): Range for random delay between requests
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
        
        # Initialize collected titles list and a requests session
        self.collected_titles = []
        self.session = requests.Session()
    
    def random_delay(self):
        """Sleep for a random amount of time."""
        delay = random.uniform(*self.delay_range)
        logger.debug(f"Waiting for {delay:.2f} seconds")
        time.sleep(delay)
    
    def extract_titles(self, html_content):
        """
        Extract research titles from HTML content using BeautifulSoup.
        
        Args:
            html_content (str): HTML content from the page
            
        Returns:
            list: List of extracted titles
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        titles = []
        
        title_elements = soup.select('h3.gs_rt')
        
        for element in title_elements:
            citation_tag = element.find('span', class_='gs_ctu')
            if citation_tag:
                citation_tag.decompose()
            
            title_link = element.find('a')
            if title_link:
                title = ' '.join(title_link.stripped_strings)
            else:
                title = ' '.join(element.stripped_strings)

            # Normalize Unicode to fix encoding issues (e.g., â€œ → “)
            title = unicodedata.normalize("NFKC", title)
            title = title.replace('\xa0', ' ')
            
            if title:
                titles.append(title)
        
        return titles
    
    def save_to_csv(self):
        """Save the collected titles to a CSV file with UTF-8 encoding."""
        try:
            with open(self.output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Title'])  # Header
                for title in self.collected_titles:
                    writer.writerow([title])
            logger.info(f"Successfully saved {len(self.collected_titles)} titles to {self.output_file}")
        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")
            raise
    
    def scrape(self, broad_query="research OR review OR paper"):
        """
        Main method to scrape Google Scholar for research titles.
        
        Args:
            broad_query (str): A broad query to use for getting results
            
        Returns:
            list: List of collected research titles
        """
        # Format the query for the URL
        formatted_query = broad_query.replace(' ', '+')
        base_url = f"https://scholar.google.com/scholar?q={formatted_query}"
        
        titles_count = 0
        current_page = 0
        
        while current_page < self.max_pages:
            if current_page == 0:
                url = base_url
            else:
                url = f"{base_url}&start={current_page * 10}"
            
            headers = {
                "User-Agent": random.choice(self.user_agents)
            }
            
            logger.info(f"Fetching page {current_page + 1}: {url}")
            try:
                response = self.session.get(url, headers=headers, timeout=10)
                if response.status_code != 200:
                    logger.warning(f"Non-200 status code received: {response.status_code}")
                    break
            except requests.RequestException as e:
                logger.error(f"Request failed: {e}")
                break
            
            page_titles = self.extract_titles(response.text)
            
            if not page_titles:
                logger.warning("No titles found on this page. Possibly blocked or reached the end.")
                break
            
            self.collected_titles.extend(page_titles)
            titles_count += len(page_titles)
            logger.info(f"Extracted {len(page_titles)} titles from page {current_page + 1}. Total: {titles_count}")
            
            current_page += 1
            
            # Random delay to avoid detection
            self.random_delay()
        
        # Save results to CSV
        self.save_to_csv()
        return self.collected_titles

def main():
    parser = argparse.ArgumentParser(description='Scrape research paper titles from Google Scholar using requests')
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
        titles = scraper.scrape(broad_query=args.query)
        print(f"Successfully scraped {len(titles)} research paper titles.")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
