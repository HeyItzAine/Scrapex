# Scrapex

Scrapex is a machine learning and web scraping project designed to automate the extraction, cleaning, and analysis of research paper titles from Google Scholar. Using Selenium and BeautifulSoup, the project collects academic research titles, processes the text with NLP techniques, and stores the structured data for further analysis.

A key feature of Scrapex is its topic modeling capability, which applies machine learning algorithms to uncover hidden themes within the collected corpus. Following MLOps principles, the project ensures a structured workflow for data collection, preprocessing, model training, and result analysis. By providing a scalable and reproducible pipeline, Scrapex helps researchers and institutions efficiently categorize research topics and gain deeper insights into academic trends.

## Data Sources
The data for this project is collected through web scraping from Google Scholar, focusing solely on research titles. These titles serve as input for topic modeling to identify recurring themes in academic research.

Source: Google Scholar 

Collected Data: Research titles

Storage: The scraped data is stored in Data/ before preprocessing

## Project Structure

```
Scrapex/
├── Data/
│   ├── research_titles.csv
│   └── research_titles_cleaned.csv
├── README.md
├── Scripts/
│   ├── Cleaner.py
│   ├── Pandas_Analysis.py
│   ├── Scrapex.py
│   ├── Text_Representation.py
│   ├── Visualization.py
│   └── requirements.txt
└── main.py 
```

## Installation and Setup

### Prerequisites
Ensure you have the following installed:
- Python 3.x
- `pip` (Python package manager)
- Google Chrome
- ChromeDriver (ensure it matches your Chrome version)

### Install Dependencies
Navigate to the `Scripts/` directory and run:
```bash
pip install -r requirements.txt
```
**Dependencies:**
- `selenium`
- `beautifulsoup4`
- `pandas`
- `nltk`
- `webdriver-manager`

## Scripts Overview

### 1. `Scrapex.py` (Web Scraper)
The `Scrapex.py` script is responsible for automating data extraction from Google Scholar. It works by launching a Chrome browser instance using Selenium, searching for research papers using a broad query, and extracting titles from multiple pages.

- **How it Works:**
  1. Initiates a Selenium WebDriver session.
  2. Navigates to Google Scholar and searches for research-related terms.
  3. Parses the HTML content using BeautifulSoup to extract paper titles.
  4. Implements user-agent rotation and random delays to prevent detection.
  5. Iterates through multiple pages using pagination.
  6. Saves the extracted data in `Data/research_titles.csv`.

#### Basic Usage:
```bash
python Scripts/Scrapex.py
```

#### Advanced Usage:
```bash
python Scripts/Scrapex.py --output "../Data/custom_output.csv" --pages 10 --query "machine learning OR artificial intelligence" --headless
```

**Advanced Options:**
- `--output`: Specify a custom output file path (default: `../Data/research_titles.csv`)
- `--pages`: Set the maximum number of pages to scrape (default: 5)
- `--query`: Define a custom search query (default: "research OR study OR thesis OR review OR paper")
- `--headless`: Run Chrome in headless mode without UI (default: True)

### 2. `Cleaner.py` (Data Cleaning Script)
The `Cleaner.py` script processes the raw research titles extracted by `Scrapex.py`. It uses NLP techniques to clean and standardize the text, making it more suitable for analysis.

- **How it Works:**
  1. Loads the raw CSV file (`research_titles.csv`).
  2. Tokenizes the text into individual words.
  3. Removes common stopwords and unwanted characters.
  4. Applies lemmatization to normalize words.
  5. Fixes character encoding issues (e.g., `â€œ` instead of `"`).
  6. Saves the cleaned data to `Data/research_titles_cleaned.csv`.

#### Basic Usage:
```bash
python Scripts/Cleaner.py
```

#### Advanced Usage:
```bash
python Scripts/Cleaner.py --input "../Data/custom_data.csv" --output "../Data/custom_data_cleaned.csv" --language "spanish"
```

**Advanced Options:**
- `--input`: Specify a custom input file path (default: `../Data/research_titles.csv`)
- `--output`: Define a custom output file path (default: `[input_name]_cleaned.csv`)
- `--language`: Set the language for NLP processing (default: "english")

## Tools & Technologies
This project integrates various tools and technologies to facilitate data processing, model training, and deployment:

- Web Scraping: BeautifulSoup, Selenium, Requests (for extracting titles from Google Scholar).
- Text Processing: NLTK (for tokenization, stopword removal, and text cleaning).

## Key Features
- Automated web scraping of research titles from Google Scholar.
- Preprocessing pipeline for text cleaning and transformation.


## Future Features
- Topic Modeling:
  - Latent Dirichlet Allocation (LDA) (gensim)
  - BERTopic (for transformer-based topic extraction)
- Data Handling & Visualization: pandas, matplotlib, seaborn, pyLDAvis.
- MLOps & Experiment Tracking: MLflow (for tracking experiments and model versions).
- Containerization: Docker (for scalable and reproducible deployment).