# Use official Python image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the application files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn pandas

# Expose FastAPI default port
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "Scripts.Scraping_Service:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
