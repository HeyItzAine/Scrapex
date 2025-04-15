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
    
    def __init__(self, input_file, output_file=None, language='english'):
        """
        Initialize the data cleaner.
        
        Args:
            input_file (str): Path to the input CSV file
            output_file (str): Path to the output CSV file (default: None)
            language (str): Language for stopwords and lemmatization (default: 'english')
        """
        self.input_file = input_file
        
        # If no output file is specified, create one with "_cleaned" suffix
        if output_file is None:
            base, ext = os.path.splitext(input_file)
            self.output_file = f"{base}_cleaned{ext}"
        else:
            self.output_file = output_file
        
        self.language = language
        
        # Download necessary NLTK resources if not already downloaded
        self._download_nltk_resources()
        
        # Initialize NLP components
        self.stopwords = set(stopwords.words(self.language))
        self.lemmatizer = WordNetLemmatizer()
        self.punctuation = set(string.punctuation)
        
    def _download_nltk_resources(self):
        """Download necessary NLTK resources."""
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
        """
        Clean a single text string using NLP techniques.
        
        Args:
            text (str): Input text to clean
        
        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and punctuation, and apply lemmatization
        clean_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stopwords and token not in self.punctuation
        ]
        
        # Join tokens back into a single string
        clean_text = ' '.join(clean_tokens)
        
        return clean_text
    
    def clean_authors(self, text):
        """
        Clean the authors' names using regex.
        
        Args:
            text (str): Input author string
        
        Returns:
            str: Cleaned author names
        """
        # Remove journal names, extra text, and trailing commas
        text = re.sub(r",?\s*-\s*[^,]+", "", text)  # Remove text after a dash
        text = re.sub(r"\.{2,}", "", text)  # Remove excessive periods
        text = re.sub(r"\s+", " ", text).strip()  # Normalize spaces
        return text
    
    def process_csv(self):
        """
        Read the input CSV, clean the data, and write to the output CSV.
        
        Returns:
            pandas.DataFrame: The cleaned data
        """
        try:
            # Read the CSV file
            logger.info(f"Reading data from {self.input_file}")
            df = pd.read_csv(self.input_file)

            # Always clean Title and Authors
            if 'Title' not in df.columns or 'Authors' not in df.columns:
                raise ValueError(f"Input CSV must contain 'Title' and 'Authors' columns. Found: {df.columns.tolist()}")
            
            logger.info("Cleaning titles and authors...")
            df['CleanedTitle'] = df['Title'].apply(self.clean_title)
            df['CleanedAuthors'] = df['Authors'].apply(self.clean_authors)

            # Optionally clean Abstract
            if 'Abstract' in df.columns:
                logger.info("Cleaning abstracts...")
                df['CleanedAbstract'] = df['Abstract'].fillna("").apply(self.clean_title)

            # Write to output file
            logger.info(f"Writing cleaned data to {self.output_file}")
            df.to_csv(self.output_file, index=False, quoting=csv.QUOTE_ALL)

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
