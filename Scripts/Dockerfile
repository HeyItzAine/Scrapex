# Use official Python image
FROM python:3.12

# Set working directory
WORKDIR /app

# Copy all project files
COPY . /app

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000 for FastAPI
EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "Scripts.Scraping_Service:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
