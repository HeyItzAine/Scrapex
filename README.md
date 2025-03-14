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
└── main.py 
```

## Installation and Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/HeyItzAine/Scrapex.git
   cd scrapex
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
Run the full pipeline:
```sh
python main.py
```

### Individual Scripts
- **Web Scraping**:
  ```sh
  python scrapex.py --query "machine learning OR artificial intelligence" --pages 10 --output "data/research_titles.csv"
  ```
- **Data Cleaning**:
  ```sh
  python cleaner.py --input "data/research_titles.csv" --output "data/cleaned_titles.csv"
  ```
- **Data Analysis**:
  ```sh
  python analyzer.py --input "data/cleaned_titles.csv"
  ```
- **Text Representation**:
  ```sh
  python text_representation.py
  ```

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

### 1. `scrapex.py` (Web Scraper)
The `scrapex.py` script automates data extraction from Google Scholar using Requests and BeautifulSoup.

- **How it Works:**
  1. Sends HTTP requests to Google Scholar using randomized user agents.
  2. Parses the HTML content with BeautifulSoup to extract research paper titles and authors.
  3. Implements random delays and user-agent rotation to reduce detection risk.
  4. Iterates through multiple pages using pagination.
  5. Saves the extracted data in `Data/research_titles.csv`.

#### Basic Usage:
```sh
python Scripts/scrapex.py
```

#### Advanced Usage:
```sh
python Scripts/scrapex.py --output "../Data/custom_output.csv" --pages 10 --query "machine learning OR artificial intelligence"
```

**Advanced Options:**
- `--output`: Specify a custom output file path (default: `../Data/research_titles.csv`)
- `--pages`: Set the maximum number of pages to scrape (default: 5)
- `--query`: Define a custom search query (default: "research OR study OR thesis OR review OR paper")

### 2. `cleaner.py` (Data Cleaning Script)
The `cleaner.py` script processes the raw research titles extracted by `scrapex.py`. It uses NLP techniques to clean and standardize the text, making it more suitable for analysis.

- **How it Works:**
  1. Loads the raw CSV file (`research_titles.csv`).
  2. Tokenizes the text into individual words.
  3. Removes common stopwords and unwanted characters.
  4. Applies lemmatization to normalize words.
  5. Fixes character encoding issues.
  6. Saves the cleaned data to `Data/research_titles_cleaned.csv`.

#### Basic Usage:
```sh
python Scripts/cleaner.py
```

#### Advanced Usage:
```sh
python Scripts/cleaner.py --input "../Data/custom_data.csv" --output "../Data/custom_data_cleaned.csv" --language "spanish"
```

**Advanced Options:**
- `--input`: Specify a custom input file path (default: `../Data/research_titles.csv`)
- `--output`: Define a custom output file path (default: `[input_name]_cleaned.csv`)
- `--language`: Set the language for NLP processing (default: "english")

## Tools & Technologies
This project integrates various tools and technologies to facilitate data processing, model training, and deployment:

- Web Scraping: Requests, BeautifulSoup (for extracting titles from Google Scholar).
- Text Processing: NLTK (for tokenization, stopword removal, and text cleaning).
- Machine Learning: Scikit-learn, Transformers, Torch (for text analysis and modeling).
- Data Visualization: Matplotlib, WordCloud.

## Key Features
- Automated web scraping of research titles from Google Scholar using Requests and BeautifulSoup.
- Preprocessing pipeline for text cleaning and transformation.
- Keyword-based research analysis.

## Future Features
- Topic Modeling:
  - Latent Dirichlet Allocation (LDA) (gensim)
  - BERTopic (for transformer-based topic extraction)
- Data Handling & Visualization: pandas, matplotlib, seaborn, pyLDAvis.
- MLOps & Experiment Tracking: MLflow (for tracking experiments and model versions).
- Containerization: Docker (for scalable and reproducible deployment).
