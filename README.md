# Scrapex: Topic Modelling
Scrapex: Topic Modelling is a machine learning project designed to perform automated topic modeling on research titles sourced from Google Scholar. By leveraging web scraping techniques, this project collects academic research titles, processes the text, and applies topic modeling algorithms to uncover hidden themes within the corpus.

The project follows MLOps principles, ensuring a structured workflow for data collection, preprocessing, model training, and result analysis. It provides a scalable and reproducible pipeline to help researchers and institutions categorize research topics efficiently.

## Data Sources
The data for this project is collected through web scraping from Google Scholar, focusing solely on research titles. These titles serve as input for topic modeling to identify recurring themes in academic research.

Source: Google Scholar
Collected Data: Research titles
Storage: The scraped data is stored in data/raw/ before preprocessing

## Directory Structure
```
Scrapex/
│── data/
│   ├── raw/              # Data mentah hasil scraping dari Google Scholar
│   ├── processed/        # Data setelah preprocessing
│   └── results/          # Hasil analisis topik
│
│── src/                  # Kode utama proyek
│   ├── scrapex.py        # Scraping dan ekstraksi data dari Google Scholar
```



## Tools & Technologies
This project integrates various tools and technologies to facilitate data processing, model training, and deployment:

- Web Scraping: BeautifulSoup, Selenium, Requests (for extracting titles from Google Scholar).
- Text Processing: NLTK, spaCy (for tokenization, stopword removal, and text cleaning).
- Topic Modeling:
  - Latent Dirichlet Allocation (LDA) (gensim)
  - BERTopic (for transformer-based topic extraction)
- Data Handling & Visualization: pandas, matplotlib, seaborn, pyLDAvis.
- MLOps & Experiment Tracking: MLflow (for tracking experiments and model versions).
- Containerization: Docker (for scalable and reproducible deployment).
## Key Features
- Automated web scraping of research titles from Google Scholar.
- Preprocessing pipeline for text cleaning and transformation.
- Topic modeling using state-of-the-art techniques.
- Experiment tracking and model versioning with MLflow.
- Scalable and reproducible architecture with Docker.
