import os
import argparse
import pandas as pd
import nltk
import string
import csv
import re
import logging
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataCleaner:
    """A class to clean research paper titles using NLP and authors using regex."""

    def _init_(self, input_file, output_file=None, language='english'):
        self.input_file = input_file
        if output_file is None:
            base, ext = os.path.splitext(input_file)
            self.output_file = f"{base}_cleaned{ext}"
        else:
            self.output_file = output_file
        self.language = language

        self._download_nltk_resources()
        self.stopwords = set(stopwords.words(self.language))
        self.lemmatizer = WordNetLemmatizer()
        self.punctuation = set(string.punctuation)

    def _download_nltk_resources(self):
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            logger.info("Downloading required NLTK resources...")
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')

    def clean_title(self, text):
        text = str(text).lower()
        tokens = word_tokenize(text)
        clean_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stopwords and token not in self.punctuation
        ]
        return ' '.join(clean_tokens)

    def clean_authors(self, text):
        text = str(text)
        text = re.sub(r",?\s*-\s*[^,]+", "", text)
        text = re.sub(r"\.{2,}", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def process_csv(self):
        try:
            logger.info(f"Reading data from {self.input_file}")
            df = pd.read_csv(self.input_file)

            if 'Title' not in df.columns or 'Authors' not in df.columns:
                raise ValueError(f"Input CSV must contain 'Title' and 'Authors' columns. Found: {df.columns.tolist()}")

            logger.info("Cleaning titles and authors...")
            df['CleanedTitle'] = df['Title'].fillna("").apply(self.clean_title)
            df['CleanedAuthors'] = df['Authors'].fillna("").apply(self.clean_authors)

            if 'Abstract' in df.columns:
                logger.info("Cleaning abstracts...")
                df['CleanedAbstract'] = df['Abstract'].fillna("").apply(self.clean_title)

            optional_columns = ['Venue', 'Publisher', 'Year', 'DOI']
            for col in optional_columns:
                if col not in df.columns:
                    logger.warning(f"Optional column '{col}' not found in input CSV, skipping.")

            columns_to_keep = ['Title', 'Authors', 'CleanedTitle', 'CleanedAuthors']
            if 'Abstract' in df.columns:
                columns_to_keep.append('CleanedAbstract')
            for col in optional_columns:
                if col in df.columns:
                    columns_to_keep.append(col)

            logger.info(f"Writing cleaned data to {self.output_file}")
            df.to_csv(self.output_file, index=False, columns=columns_to_keep, quoting=csv.QUOTE_ALL)

            logger.info(f"Successfully cleaned {len(df)} records")
            return df

        except Exception as e:
            logger.error(f"Error during data cleaning: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Clean research paper titles using NLP and authors using regex')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file path (default: input_cleaned.csv)')
    parser.add_argument('--language', type=str, default='english', help='Language for NLP processing')

    args = parser.parse_args()

    cleaner = DataCleaner(
        input_file=args.input,
        output_file=args.output,
        language=args.language
    )

    try:
        cleaner.process_csv()
        print(f"Successfully cleaned data and saved to {cleaner.output_file}")
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())