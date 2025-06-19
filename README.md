# Scrapex

Scrapex is a machine learning and web scraping project designed to automate the extraction, cleaning, and analysis of research paper titles from Google Scholar. Using FastAPI, Requests, and BeautifulSoup, Scrapex collects academic research titles, processes the text with NLP techniques, and provides an API for accessing the data.

A key feature of Scrapex is its **topic modeling capability**, applying ML algorithms to uncover hidden themes in the collected corpus. Following **MLOps principles**, Scrapex ensures a structured workflow for **data collection, preprocessing, model training, and deployment** via **Docker**.

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
│   ├── research_titles_cleaned.csv
│   ├── semantic_titles.csv
│   └── semantic_titles_cleaned.csv
├── Dockerfile
├── README.md
├── Scripts/
│   ├── BERTopic.py
│   ├── Cleaner.py
│   ├── Converter.py
│   ├── Pandas_Analysis.py
│   ├── Result_Service.py
│   ├── Scrapex.py
│   ├── Scraping_Service.py
│   ├── Semantic_Scrapex.py
│   ├── Text_Representation.py
│   ├── Visualization.py
│   ├── mlflow_example.py
│   ├── mlflow_experiment.py
│   ├── scrapex_mlflow_simple.py
│   └── requirements.txt
├── docker-compose.yml
├── mlruns/
├── research_titles_cleaned_BERTopic/
├── semantic_titles_cleaned_BERTopic/
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

## Running the FastAPI Server

Scrapex now includes an **API for fetching, searching, and analyzing scraped research titles**.

### 1. Start the FastAPI Server
```sh
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The server provides two main endpoints:
- `/start-scraping/` - Triggers the scraping process in the background
- `/scraped-data/` - Returns the cleaned research data in JSON format

### 2. API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/start-scraping/` | Start a new web scraping job in the background |
| `GET` | `/scraped-data/` | Get all scraped and cleaned research papers |

### 3. Access the API Docs

Once the FastAPI server is running, open:
- **Swagger UI**: http://localhost:8000/docs
- **Redoc UI**: http://localhost:8000/redoc

## Running Scrapex with Docker

Scrapex is containerized using **Docker**, ensuring easy deployment and reproducibility.

### 1. Build the Docker Image
```sh
docker build -t scrapex .
```

### 2. Run the Docker Container
```sh
docker run -p 8000:8000 scrapex
```

This will:
* Start the **FastAPI API** inside the container
* Scrape and store research titles
* Expose the API at http://localhost:8000

### 3. Run with Docker Compose
```sh
docker-compose up
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

### 3. `BERTopic.py` (Topic Modeling)
The `BERTopic.py` script applies topic modeling to cleaned research titles using BERTopic, a transformer-based topic modeling technique.

- **How it Works:**
  1. Loads the cleaned titles from CSV files.
  2. Creates a BERTopic model with specified parameters.
  3. Transforms text data into topics.
  4. Generates human-readable topic names.
  5. Saves the model, topics, and visualizations.

#### Basic Usage:
```sh
python Scripts/BERTopic.py Data/research_titles_cleaned.csv all
```

#### Advanced Options:
- First argument: Path to the cleaned CSV file
- Second argument: Topic query (use "all" for all topics or specific topic number)

## Experiment Tracking and Monitoring

### MLflow Tracking
Scrapex uses **MLflow** for experiment tracking. All model runs, parameters, and metrics are logged to the `mlruns/` directory. You can view the MLflow UI by running:

```sh
python Scripts/mlflow_experiment.py --no-ui  # To run pipeline without UI
mlflow ui --port 5000                        # To launch the MLflow UI
```

Open [http://localhost:5000](http://localhost:5000) in your browser to explore experiment results.

### Monitoring with Prometheus and Grafana
Scrapex exposes Prometheus metrics for monitoring scraping jobs. Metrics include request counts, durations, exceptions, and last scrape time. To enable monitoring:

1. **Start Prometheus** using the provided `prometheus.yml` configuration.
2. **Start Grafana** and import the dashboard from `grafana/dashboard.json`.
3. The dashboard visualizes scraping duration, success/failure rates, status distribution, and last scrape time.

#### Example Prometheus Metrics
- `scrapex_requests_total{status="success"}`: Number of successful requests
- `scrapex_requests_total{status="failure"}`: Number of failed requests
- `scrapex_request_duration_seconds`: Scraping duration histogram
- `scrapex_last_scrape_unixtime`: Last scrape timestamp

#### Example Grafana Panels
- **Durasi Scraping**: Time series of scraping durations
- **Error vs Sukses**: Success vs failure rates
- **Distribusi Status Scraping**: Pie chart of request status
- **Waktu Terakhir Scraping**: Last scrape time

## Data Version Control (DVC)
Scrapex recommends using **DVC** for data versioning and reproducibility. DVC tracks changes to datasets and model artifacts, enabling consistent experiments.

### Basic DVC Workflow
1. **Initialize DVC** (run once):
   ```sh
   dvc init
   ```
2. **Track a data file**:
   ```sh
   dvc add Data/research_titles.csv
   dvc add Data/semantic_titles.csv
   ```
3. **Commit DVC files**:
   ```sh
   git add Data/*.dvc .gitignore
   git commit -m "Track data with DVC"
   ```
4. **Push data to remote storage** (optional):
   ```sh
   dvc remote add -d myremote <remote-url>
   dvc push
   ```

For more details, see the [DVC documentation](https://dvc.org/doc/start).

## Tools & Technologies
This project integrates various tools and technologies to facilitate data processing, model training, and deployment:

- Web Scraping: Requests, BeautifulSoup (for extracting titles from Google Scholar).
- Text Processing: NLTK (for tokenization, stopword removal, and text cleaning).
- Machine Learning: Scikit-learn, Transformers, Torch (for text analysis and modeling).
- Data Visualization: Matplotlib, WordCloud.
- Docker Containerization
- FastAPI
- Experiment Tracking: MLflow (for tracking machine learning experiments)
- Topic Modeling: BERTopic (for transformer-based topic discovery)

## Key Features
- Automated web scraping of research titles from Google Scholar  
- FastAPI-powered API for querying scraped data  
- Docker containerization for easy deployment  
- Text cleaning & NLP preprocessing  
- Topic modeling with BERTopic for insight discovery
- MLflow experiment tracking for model comparison

## Future Features
- Topic Modeling:
  - Latent Dirichlet Allocation (LDA) (gensim)
  - BERTopic (for transformer-based topic extraction)
- Data Handling & Visualization: pandas, matplotlib, seaborn, pyLDAvis.
- MLOps & Experiment Tracking: MLflow (for tracking experiments and model versions).
- Containerization: Docker (for scalable and reproducible deployment).
